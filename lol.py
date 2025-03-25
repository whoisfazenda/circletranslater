import telebot
from telebot import types
from vosk import Model, KaldiRecognizer
import wave
from pydub import AudioSegment
import subprocess
import requests
from googletrans import Translator  # Импортируем googletrans
import ffmpeg
import os
import logging
import time  # Импорт time для повторных попыток
from langdetect import detect, DetectorFactory
from langdetect.lang_detect_exception import LangDetectException
import translate_audio_video  # Импортируем функции перевода
from moviepy import VideoFileClip, vfx  # Импортируем moviepy для обработки видео
import fasttext  # Импортируем fasttext для определения языка




# Устанавливаем seed для воспроизводимости результатов langdetect
DetectorFactory.seed = 0
from langdetect import detect_langs
detect_langs("test")

# Настройка логирования
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Загружаем модель fasttext для определения языка
try:
    language_detector = fasttext.load_model("lid.176.bin")
    logging.info("Модель fasttext успешно загружена")
except Exception as e:
    logging.error(f"Не удалось загрузить модель fasttext: {e}. Убедись, что файл lid.176.bin находится в папке проекта.")
    exit(1)

# Токен Telegram бота
bot = telebot.TeleBot("8041048168:AAE6Mi8o0bZppdsvGnQypykF0zXLtzduoJs")

# Инициализация клиента Google Translate (googletrans)
translator = Translator()

# Пути к моделям Vosk
VOSK_MODEL_PATHS = {
    "ru": "vosk-model-small-ru-0.22",
    "en": "vosk-model-small-en-us-zamia-0.5",
    "uk": "vosk-model-small-uk-v3-nano"
}

# Инициализация моделей Vosk
vosk_models = {}
for lang, path in VOSK_MODEL_PATHS.items():
    if not os.path.exists(path):
        logging.error(f"Модель Vosk для языка {lang} не найдена по пути: {path}. Скачай модель с https://alphacephei.com/vosk/models")
        exit(1)
    vosk_models[lang] = Model(path)

# Словарь для хранения временных данных
pending_actions = {}
processed_messages = set()  # Для предотвращения дублирования обработки сообщений

# Функция для создания главного меню
def create_main_menu():
    markup = types.ReplyKeyboardMarkup(resize_keyboard=True)
    btn1 = types.KeyboardButton("🎶 Расшифровать аудио")
    btn2 = types.KeyboardButton("📹 Расшифровать видео")
    btn3 = types.KeyboardButton("📹➡️⭕️ Конвертировать видео в кружочек")
    btn4 = types.KeyboardButton("👅 Перевести аудио/видео на другой язык")
    markup.add(btn1, btn2)
    markup.add(btn3, btn4)
    return markup

# Функция для безопасного удаления файла с повторными попытками
def safe_remove(file_path, retries=5, delay=1):
    for attempt in range(retries):
        try:
            if os.path.exists(file_path):
                os.remove(file_path)
                logging.info(f"Файл удалён: {file_path}")
            return True
        except Exception as e:
            logging.warning(f"Не удалось удалить файл {file_path} (попытка {attempt + 1}/{retries}): {e}")
            time.sleep(delay)
    logging.error(f"Не удалось удалить файл после {retries} попыток: {file_path}")
    return False

# Команда /start
@bot.message_handler(commands=['start'])
def send_welcome(message):
    try:
        logging.info(f"Получена команда /start от пользователя {message.chat.id}")
        bot.reply_to(message, "🖐 Привет! Я могу перевести твой кружочек на родной тебе язык) \nВыбери действие из меню:", reply_markup=create_main_menu())
    except Exception as e:
        logging.error(f"Ошибка в send_welcome: {e}")

# Функция извлечения аудио из видео или голосового сообщения
def extract_audio(file_path, file_type="video"):
    logging.info(f"Извлечение аудио из {file_type}: {file_path}")
    try:
        audio = AudioSegment.from_file(file_path, format="mp4" if file_type == "video" else "ogg")
        audio = audio.set_channels(1).set_frame_rate(16000).set_sample_width(2)
        audio.export("original_audio.wav", format="wav")
        duration_ms = len(audio)
        return "original_audio.wav", duration_ms / 1000
    except Exception as e:
        logging.error(f"❌ Ошибка при извлечении аудио: {e}")
        return None, 0

# Функция распознавания речи с определением языка через fasttext
def recognize_speech(audio_path):
    logging.info(f"Распознавание речи из файла: {audio_path}")
    try:
        wf = wave.open(audio_path, "rb")
        if wf.getnchannels() != 1 or wf.getsampwidth() != 2 or wf.getframerate() != 16000:
            logging.error("Аудиофайл должен быть WAV, моно, 16-bit, 16000 Hz")
            return "❌ Ошибка: неподходящий формат аудио"

        # Предварительное распознавание для определения языка (используем русскую модель для первого прохода)
        text_for_detection = ""
        recognizer_temp = KaldiRecognizer(vosk_models["ru"], wf.getframerate())
        while True:
            data = wf.readframes(4000)
            if len(data) == 0:
                break
            if recognizer_temp.AcceptWaveform(data):
                result = recognizer_temp.Result()
                result_dict = eval(result.replace("true", "True").replace("false", "False"))
                text_for_detection += result_dict.get("text", "") + " "
        final_result = recognizer_temp.FinalResult()
        final_dict = eval(final_result.replace("true", "True").replace("false", "False"))
        text_for_detection += final_dict.get("text", "")
        text_for_detection = text_for_detection.strip()

        # Определяем язык с помощью fasttext
        detected_lang = None
        if text_for_detection:
            predictions = language_detector.predict(text_for_detection.replace("\n", " "))
            detected_lang = predictions[0][0].replace("__label__", "")
            if detected_lang not in ["ru", "en", "uk"]:
                detected_lang = None
            logging.info(f"Определённый язык (fasttext): {detected_lang}, вероятность: {predictions[1][0]}")
        else:
            logging.warning("Не удалось получить текст для определения языка")

        # Определяем порядок языков для распознавания
        languages = ["ru", "en", "uk"]
        if detected_lang:
            languages = [detected_lang] + [lang for lang in languages if lang != detected_lang]
        logging.info(f"Порядок языков для распознавания: {languages}")

        # Распознаём речь
        for lang in languages:
            logging.info(f"Попытка распознавания на языке: {lang}")
            wf.rewind()
            recognizer = KaldiRecognizer(vosk_models[lang], wf.getframerate())
            recognizer.SetWords(True)

            text = ""
            while True:
                data = wf.readframes(4000)
                if len(data) == 0:
                    break
                if recognizer.AcceptWaveform(data):
                    result = recognizer.Result()
                    result_dict = eval(result.replace("true", "True").replace("false", "False"))
                    text += result_dict.get("text", "") + " "

            final_result = recognizer.FinalResult()
            final_dict = eval(final_result.replace("true", "True").replace("false", "False"))
            text += final_dict.get("text", "")
            text = text.strip()

            if text:
                logging.info(f"Распознанный текст (Vosk, {lang}): {text}")
                return text

        logging.warning("Речь не распознана ни на одном языке")
        return "❌ Не удалось распознать речь"
    except Exception as e:
        logging.error(f"Ошибка при распознавании речи: {e}")
        return "❌ Ошибка при распознавании речи"

# Функция конвертации видео в кружочек (уменьшаем битрейт и исправляем subclip)
def convert_to_circle(video_path, output_path="circle_video.mp4"):
    logging.info(f"Конвертация видео в кружочек: {video_path}")
    try:
        # Получаем информацию о видео
        probe = ffmpeg.probe(video_path)
        video_stream = next(s for s in probe['streams'] if s['codec_type'] == 'video')
        width = int(video_stream['width'])
        height = int(video_stream['height'])
        duration = float(probe['format']['duration'])

        # Ограничиваем длительность до 60 секунд
        duration = min(duration, 60)
        logging.info(f"Длительность после проверки: {duration} сек")

        # Определяем размер для квадратного видео
        size = min(width, height)
        logging.info(f"Размер после обрезки: {size}x{size}")

        # Проверяем наличие аудио
        has_audio = any(stream['codec_type'] == 'audio' for stream in probe['streams'])
        logging.info(f"Видео имеет аудиопоток: {has_audio}")

        # Используем ffmpeg для обрезки, масштабирования и конвертации
        cmd = [
            "ffmpeg",
            "-i", video_path,
            "-t", str(duration),  # Ограничиваем длительность
            "-vf", f"crop={size}:{size}:{(width-size)/2}:{(height-size)/2},scale=384:384",  # Обрезаем и масштабируем
            "-c:v", "libx264",  # Кодек видео
            "-b:v", "300k",  # Битрейт видео
            "-r", "24",  # Частота кадров
        ]

        if has_audio:
            cmd.extend([
                "-c:a", "aac",  # Кодек аудио
                "-b:a", "64k"  # Битрейт аудио
            ])
        else:
            cmd.append("-an")  # Убираем аудио, если его нет

        cmd.extend([
            "-y",  # Перезаписываем выходной файл
            output_path
        ])

        # Выполняем команду
        subprocess.run(cmd, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        logging.info(f"Кружочек готов: {output_path}")

        # Проверяем размер файла
        file_size = os.path.getsize(output_path) / (1024 * 1024)  # Размер в МБ
        logging.info(f"Размер итогового файла: {file_size:.2f} МБ")
        if file_size > 10:
            logging.warning("Размер файла превышает 10 МБ, Telegram может отклонить video_note")

        # Проверяем параметры итогового файла
        probe_output = ffmpeg.probe(output_path)
        has_audio_output = any(stream['codec_type'] == 'audio' for stream in probe_output['streams'])
        logging.info(f"Итоговый файл имеет аудиопоток: {has_audio_output}")

        video_info_output = next(s for s in probe_output['streams'] if s['codec_type'] == 'video')
        logging.info(f"Параметры итогового видео: размер={video_info_output['width']}x{video_info_output['height']}, кодек={video_info_output['codec_name']}, fps={video_info_output.get('r_frame_rate')}")

        return output_path
    except Exception as e:
        logging.error(f"❌ Ошибка при конвертации видео в кружочек: {str(e)}")
        return None

# Обработка текстовых сообщений (для кнопок)
@bot.message_handler(content_types=['text'])
def handle_text(message):
    try:
        user_id = message.chat.id
        text = message.text.strip()  # Убираем лишние пробелы
        logging.info(f"Получено текстовое сообщение от пользователя {user_id}: '{text}'")

        if text == "🎶 Расшифровать аудио":
            logging.info(f"Пользователь {user_id} нажал 'Расшифровать аудио'")
            pending_actions[user_id] = {"type": "transcribe_audio"}
            bot.reply_to(message, "🗣 Отправь голосовое сообщение для расшифровки.")
        elif text == "📹 Расшифровать видео":
            logging.info(f"Пользователь {user_id} нажал 'Расшифровать видео'")
            pending_actions[user_id] = {"type": "transcribe_video"}
            bot.reply_to(message, "📹 Отправь видео или кружочек для расшифровки.")
        elif text == "📹➡️⭕️ Конвертировать видео в кружочек":
            logging.info(f"Пользователь {user_id} нажал 'Конвертировать видео в кружочек'")
            pending_actions[user_id] = {"type": "convert_to_circle"}
            bot.reply_to(message, "📹➡️⭕️ Отправь видео для конвертации в кружочек.")
        elif text == "👅 Перевести аудио/видео на другой язык":
            logging.info(f"Пользователь {user_id} нажал 'Перевести аудио/видео на другой язык'")
            pending_actions[user_id] = {"type": "translate_audio_video"}
            bot.reply_to(message, "👅 Отправь голосовое сообщение или кружочек для перевода.")
        else:
            logging.info(f"Неизвестный текст от пользователя {user_id}: '{text}'")
            bot.reply_to(message, "Пожалуйста, выберите действие из меню:", reply_markup=create_main_menu())
    except Exception as e:
        logging.error(f"Ошибка в handle_text: {e}")

# Обработка голосовых сообщений
@bot.message_handler(content_types=['voice'])
def handle_voice(message):
    try:
        user_id = message.chat.id
        message_id = message.message_id
        if message_id in processed_messages:
            logging.info(f"Сообщение {message_id} уже обработано, пропускаем")
            return
        processed_messages.add(message_id)

        logging.info(f"Получено голосовое сообщение от пользователя {user_id}")

        if user_id not in pending_actions:
            logging.warning(f"Голосовое сообщение от {user_id} не ожидалось (pending_actions: {pending_actions.get(user_id)})")
            bot.reply_to(message, "Сначала выбери действие из меню.", reply_markup=create_main_menu())
            return

        if pending_actions[user_id]["type"] == "transcribe_audio":
            transcribe_audio_handler(message)
        elif pending_actions[user_id]["type"] == "translate_audio_video":
            translate_audio_video.handle_voice(bot, message, extract_audio, recognize_speech, create_main_menu)
        else:
            logging.warning(f"Голосовое сообщение от {user_id} не соответствует ожидаемому действию (pending_actions: {pending_actions.get(user_id)})")
            bot.reply_to(message, "Сначала выбери действие из меню.", reply_markup=create_main_menu())
    except Exception as e:
        logging.error(f"Ошибка в handle_voice: {e}")

# Обработка кружочков
@bot.message_handler(content_types=['video_note'])
def handle_video_note(message):
    try:
        user_id = message.chat.id
        message_id = message.message_id
        if message_id in processed_messages:
            logging.info(f"Сообщение {message_id} уже обработано, пропускаем")
            return
        processed_messages.add(message_id)

        logging.info(f"Получен кружочек от пользователя {user_id}")

        if user_id not in pending_actions:
            logging.warning(f"Кружочек от {user_id} не ожидался (pending_actions: {pending_actions.get(user_id)})")
            bot.reply_to(message, "Сначала выбери действие из меню.", reply_markup=create_main_menu())
            return

        if pending_actions[user_id]["type"] == "translate_audio_video":
            translate_audio_video.handle_video_note(bot, message, extract_audio, recognize_speech, create_main_menu)
        elif pending_actions[user_id]["type"] == "transcribe_video":
            # Обрабатываем кружочек как видео для расшифровки
            file_id = message.video_note.file_id
            file_info = bot.get_file(file_id)
            downloaded_file = bot.download_file(file_info.file_path)
            video_path = f"video_note_{user_id}.mp4"
            with open(video_path, "wb") as video_file:
                video_file.write(downloaded_file)

            audio_path, _ = extract_audio(video_path, file_type="video")
            if not audio_path:
                bot.reply_to(message, "❌ Ошибка при извлечении аудио из кружочка.", reply_markup=create_main_menu())
                return

            text = recognize_speech(audio_path)
            if "Ошибка" in text or "Не удалось" in text:
                bot.reply_to(message, text, reply_markup=create_main_menu())
            else:
                bot.reply_to(message, f"Расшифрованный текст: {text}", reply_markup=create_main_menu())

            safe_remove(video_path)
            safe_remove(audio_path)
            if user_id in pending_actions:
                del pending_actions[user_id]
        else:
            logging.warning(f"Кружочек от {user_id} не соответствует ожидаемому действию (pending_actions: {pending_actions.get(user_id)})")
            bot.reply_to(message, "Сначала выбери действие из меню.", reply_markup=create_main_menu())
    except Exception as e:
        logging.error(f"Ошибка в handle_video_note: {e}")

# Обработка видео
@bot.message_handler(content_types=['video'])
def handle_video(message):
    try:
        user_id = message.chat.id
        message_id = message.message_id
        if message_id in processed_messages:
            logging.info(f"Сообщение {message_id} уже обработано, пропускаем")
            return
        processed_messages.add(message_id)

        logging.info(f"Получено видео от пользователя {user_id}")

        if user_id not in pending_actions:
            logging.warning(f"Видео от {user_id} не ожидалось (pending_actions: {pending_actions.get(user_id)})")
            bot.reply_to(message, "Сначала выбери действие из меню.", reply_markup=create_main_menu())
            return

        if pending_actions[user_id]["type"] == "transcribe_video":
            transcribe_video_handler(message)
        elif pending_actions[user_id]["type"] == "convert_to_circle":
            convert_video_to_circle_handler(message)
        else:
            logging.warning(f"Видео от {user_id} не соответствует ожидаемому действию (pending_actions: {pending_actions.get(user_id)})")
            bot.reply_to(message, "Сначала выбери действие из меню.", reply_markup=create_main_menu())
    except Exception as e:
        logging.error(f"Ошибка в handle_video: {e}")

# Обработка голосовых сообщений для расшифровки
def transcribe_audio_handler(message):
    try:
        user_id = message.chat.id
        logging.info(f"Получено голосовое сообщение для расшифровки от пользователя {user_id}")

        file_info = bot.get_file(message.voice.file_id)
        downloaded_file = bot.download_file(file_info.file_path)
        voice_path = f"voice_{user_id}.ogg"
        with open(voice_path, "wb") as voice_file:
            voice_file.write(downloaded_file)

        audio_path, _ = extract_audio(voice_path, file_type="voice")
        if not audio_path:
            bot.reply_to(message, "❌ Ошибка при извлечении аудио из голосового сообщения.", reply_markup=create_main_menu())
            return

        text = recognize_speech(audio_path)
        if "Ошибка" in text or "Не удалось" in text:
            bot.reply_to(message, text, reply_markup=create_main_menu())
        else:
            bot.reply_to(message, f"📩 Расшифрованный текст: {text}", reply_markup=create_main_menu())

        # Безопасное удаление файлов
        safe_remove(voice_path)
        safe_remove(audio_path)
        if user_id in pending_actions:
            del pending_actions[user_id]
    except Exception as e:
        logging.error(f"Ошибка в transcribe_audio_handler: {e}")
        bot.reply_to(message, "❌ Произошла ошибка при обработке голосового сообщения.", reply_markup=create_main_menu())

# Обработка видео для расшифровки
def transcribe_video_handler(message):
    try:
        user_id = message.chat.id
        logging.info(f"Получено видео/кружочек для расшифровки от пользователя {user_id}")

        file_id = message.video.file_id if message.content_type == 'video' else message.video_note.file_id
        file_info = bot.get_file(file_id)
        downloaded_file = bot.download_file(file_info.file_path)
        video_path = f"video_{user_id}.mp4"
        with open(video_path, "wb") as video_file:
            video_file.write(downloaded_file)

        audio_path, _ = extract_audio(video_path, file_type="video")
        if not audio_path:
            bot.reply_to(message, "❌ Ошибка при извлечении аудио из видео.", reply_markup=create_main_menu())
            return

        text = recognize_speech(audio_path)
        if "Ошибка" in text or "Не удалось" in text:
            bot.reply_to(message, text, reply_markup=create_main_menu())
        else:
            bot.reply_to(message, f"📩 Расшифрованный текст: {text}", reply_markup=create_main_menu())

        # Безопасное удаление файлов
        safe_remove(video_path)
        safe_remove(audio_path)
        if user_id in pending_actions:
            del pending_actions[user_id]
    except Exception as e:
        logging.error(f"Ошибка в transcribe_video_handler: {e}")
        bot.reply_to(message, "❌ Произошла ошибка при обработке видео.", reply_markup=create_main_menu())

# Обработка видео для конвертации в кружочек
def convert_video_to_circle_handler(message):
    try:
        user_id = message.chat.id
        logging.info(f"Получено видео для конвертации в кружочек от пользователя {user_id}")

        file_info = bot.get_file(message.video.file_id)
        downloaded_file = bot.download_file(file_info.file_path)
        video_path = f"video_{user_id}.mp4"
        with open(video_path, "wb") as video_file:
            video_file.write(downloaded_file)

        output_path = convert_to_circle(video_path)
        if output_path:
            with open(output_path, "rb") as video:
                # Отправляем кружочек и сохраняем message_id
                sent_message = bot.send_video_note(user_id, video, reply_markup=create_main_menu())
                processed_messages.add(sent_message.message_id)  # Добавляем message_id в processed_messages
            safe_remove(video_path)
            safe_remove(output_path)
        else:
            bot.reply_to(message, "❌ Произошла ошибка при конвертации видео в кружочек.", reply_markup=create_main_menu())

        if user_id in pending_actions:
            del pending_actions[user_id]
    except Exception as e:
        logging.error(f"Ошибка в convert_video_to_circle_handler: {e}")
        bot.reply_to(message, "❌ Произошла ошибка при конвертации видео в кружочек.", reply_markup=create_main_menu())

# Обработка выбора пола и языка через инлайн-кнопки
@bot.callback_query_handler(func=lambda call: True)
def callback_inline(call):
    try:
        translate_audio_video.callback_inline(bot, call, create_main_menu)
    except Exception as e:
        logging.error(f"Ошибка в callback_inline: {e}")

# Запуск бота с перезапуском при ошибках
def run_bot():
    while True:
        try:
            logging.info("Запускаю бота...")
            bot.polling(none_stop=True, interval=0, timeout=20)
        except Exception as e:
            logging.error(f"Ошибка в bot.polling, перезапускаю бота через 5 секунд: {e}")
            time.sleep(5)  # Задержка перед перезапуском

if __name__ == "__main__":
    run_bot()