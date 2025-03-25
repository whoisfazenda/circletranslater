import telebot
from telebot import types
from gtts import gTTS
from googletrans import Translator
import os
import logging
import ffmpeg
import time
from pydub import AudioSegment
from langdetect import detect
import fasttext
from vosk import Model, KaldiRecognizer
import wave
import json

# Настройка логирования
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Инициализация клиента Google Translate
translator = Translator()

# Загружаем модель fasttext для определения языка
language_model = fasttext.load_model("lid.176.bin")

# Путь к моделям Vosk
VOSK_MODELS = {
    "ru": "vosk-model-small-ru-0.22",
    "en": "vosk-model-small-en-us-zamia-0.5",
    "uk": "vosk-model-small-uk-v3-nano"
}

# Словарь для хранения данных о переводе
translation_data = {}

# Функция для создания клавиатуры выбора пола
def create_gender_keyboard():
    markup = types.InlineKeyboardMarkup()
    btn_male = types.InlineKeyboardButton("Мужской голос", callback_data="gender_male")
    btn_female = types.InlineKeyboardButton("Женский голос", callback_data="gender_female")
    markup.add(btn_male, btn_female)
    return markup

# Функция для создания клавиатуры выбора языка перевода
def create_language_keyboard():
    markup = types.InlineKeyboardMarkup()
    btn_en = types.InlineKeyboardButton("Английский", callback_data="lang_en")
    btn_ru = types.InlineKeyboardButton("Русский", callback_data="lang_ru")
    btn_uk = types.InlineKeyboardButton("Украинский", callback_data="lang_uk")
    markup.add(btn_en, btn_ru, btn_uk)
    return markup

# Функция для создания клавиатуры с опциями после распознавания (выбор пола + изменение языка)
def create_recognition_options_keyboard():
    markup = types.InlineKeyboardMarkup()
    btn_male = types.InlineKeyboardButton("Мужской голос", callback_data="gender_male")
    btn_female = types.InlineKeyboardButton("Женский голос", callback_data="gender_female")
    btn_change_lang = types.InlineKeyboardButton("Изменить язык", callback_data="change_lang")
    markup.add(btn_male, btn_female)
    markup.add(btn_change_lang)
    return markup

# Функция для создания клавиатуры выбора языка распознавания
def create_recognition_language_keyboard():
    markup = types.InlineKeyboardMarkup()
    btn_en = types.InlineKeyboardButton("Английский", callback_data="recog_lang_en")
    btn_ru = types.InlineKeyboardButton("Русский", callback_data="recog_lang_ru")
    btn_uk = types.InlineKeyboardButton("Украинский", callback_data="recog_lang_uk")
    markup.add(btn_en, btn_ru, btn_uk)
    return markup

# Функция распознавания речи с возможностью принудительного выбора языка
def recognize_speech(audio_path, forced_lang=None):
    try:
        # Открываем аудиофайл
        wf = wave.open(audio_path, "rb")
        if wf.getnchannels() != 1 or wf.getsampwidth() != 2 or wf.getframerate() not in [16000, 44100]:
            logging.error("Аудиофайл должен быть в формате WAV, моно, 16 бит, с частотой 16000 или 44100 Гц")
            return "Ошибка: неподходящий формат аудио."

        # Если язык принудительно указан, используем его
        if forced_lang and forced_lang in VOSK_MODELS:
            detected_lang = forced_lang
            logging.info(f"Используем принудительно указанный язык: {detected_lang}")
        else:
            # Читаем аудиоданные для предварительного определения языка
            audio_data = wf.readframes(wf.getnframes())
            wf.rewind()

            # Конвертируем аудио в текст для определения языка (грубый подход)
            temp_model = Model(VOSK_MODELS["en"])  # Сначала пробуем английскую модель
            temp_rec = KaldiRecognizer(temp_model, wf.getframerate())
            temp_rec.SetWords(True)

            while True:
                data = wf.readframes(4000)
                if len(data) == 0:
                    break
                temp_rec.AcceptWaveform(data)

            temp_result = json.loads(temp_rec.FinalResult())
            temp_text = temp_result.get("text", "")
            logging.info(f"Временный распознанный текст для определения языка: {temp_text}")

            # Определяем язык с помощью fasttext
            if temp_text:
                predictions = language_model.predict(temp_text)
                detected_lang = predictions[0][0].replace("__label__", "")
                confidence = predictions[1][0]
                logging.info(f"Определённый язык: {detected_lang} (уверенность: {confidence})")
            else:
                detected_lang = "en"  # По умолчанию английский, если текст не распознан
                confidence = 0.0
                logging.warning("Не удалось распознать текст для определения языка, используем английский по умолчанию")

            # Если уверенность низкая, пробуем другую модель
            if confidence < 0.7:
                logging.info("Низкая уверенность в определении языка, пробуем другую модель")
                wf.rewind()
                temp_model = Model(VOSK_MODELS["ru"])  # Пробуем русскую модель
                temp_rec = KaldiRecognizer(temp_model, wf.getframerate())
                temp_rec.SetWords(True)

                while True:
                    data = wf.readframes(4000)
                    if len(data) == 0:
                        break
                    temp_rec.AcceptWaveform(data)

                temp_result = json.loads(temp_rec.FinalResult())
                temp_text = temp_result.get("text", "")
                predictions = language_model.predict(temp_text)
                detected_lang = predictions[0][0].replace("__label__", "")
                confidence = predictions[1][0]
                logging.info(f"Повторное определение языка: {detected_lang} (уверенность: {confidence})")

        # Выбираем модель Vosk на основе определённого языка
        if detected_lang not in VOSK_MODELS:
            detected_lang = "en"  # По умолчанию английский, если язык не поддерживается
            logging.warning(f"Язык {detected_lang} не поддерживается, используем английский")

        model = Model(VOSK_MODELS[detected_lang])
        rec = KaldiRecognizer(model, wf.getframerate())
        rec.SetWords(True)

        # Распознаём речь с правильной моделью
        wf.rewind()
        while True:
            data = wf.readframes(4000)
            if len(data) == 0:
                break
            rec.AcceptWaveform(data)

        result = json.loads(rec.FinalResult())
        text = result.get("text", "")
        if not text:
            return "Не удалось распознать речь."

        logging.info(f"Распознанный текст: {text} (язык: {detected_lang})")
        return text, detected_lang

    except Exception as e:
        logging.error(f"Ошибка при распознавании речи: {e}")
        return f"Ошибка при распознавании речи: {str(e)}", None

# Обработка голосовых сообщений
def handle_voice(bot, message, extract_audio, recognize_speech, create_main_menu):
    try:
        user_id = message.chat.id
        logging.info(f"Получено голосовое сообщение для перевода от пользователя {user_id}")

        # Извлекаем аудио из голосового сообщения
        file_info = bot.get_file(message.voice.file_id)
        downloaded_file = bot.download_file(file_info.file_path)
        voice_path = f"voice_{user_id}.ogg"
        with open(voice_path, "wb") as voice_file:
            voice_file.write(downloaded_file)

        audio_path, duration = extract_audio(voice_path, file_type="voice")
        if not audio_path:
            bot.reply_to(message, "Ошибка при извлечении аудио из голосового сообщения.", reply_markup=create_main_menu())
            return

        # Распознаём речь
        result = recognize_speech(audio_path)
        text, detected_lang = result if isinstance(result, tuple) else (result, None)
        if "Ошибка" in text or "Не удалось" in text:
            bot.reply_to(message, text, reply_markup=create_main_menu())
            return

        if not detected_lang:
            # Если язык не определён, используем langdetect как запасной вариант
            detected_lang = detect(text)
            logging.info(f"Язык определён через langdetect: {detected_lang}")

        logging.info(f"Определённый язык текста: {detected_lang}")

        # Сохраняем данные для возможного повторного распознавания
        translation_data[user_id] = {
            "text": text,
            "voice_path": voice_path,
            "audio_path": audio_path,
            "duration": duration,
            "source_lang": detected_lang
        }

        # Показываем распознанный текст и предлагаем выбрать пол или изменить язык
        bot.reply_to(message, f"Распознанный текст: {text}\nЯзык: {detected_lang}\nВыбери пол голоса для перевода или измени язык:", reply_markup=create_recognition_options_keyboard())

    except Exception as e:
        logging.error(f"Ошибка в handle_voice: {e}")
        bot.reply_to(message, "❌ Произошла ошибка при обработке голосового сообщения.", reply_markup=create_main_menu())

# Обработка кружочков
def handle_video_note(bot, message, extract_audio, recognize_speech, create_main_menu):
    try:
        user_id = message.chat.id
        logging.info(f"Получен кружочек для перевода от пользователя {user_id}")

        file_info = bot.get_file(message.video_note.file_id)
        downloaded_file = bot.download_file(file_info.file_path)
        video_path = f"video_note_{user_id}.mp4"
        with open(video_path, "wb") as video_file:
            video_file.write(downloaded_file)

        audio_path, duration = extract_audio(video_path, file_type="video")
        if not audio_path:
            bot.reply_to(message, "❌ Ошибка при извлечении аудио из кружочка.", reply_markup=create_main_menu())
            return

        # Распознаём речь
        result = recognize_speech(audio_path)
        text, detected_lang = result if isinstance(result, tuple) else (result, None)
        if "Ошибка" in text or "Не удалось" in text:
            bot.reply_to(message, text, reply_markup=create_main_menu())
            return

        if not detected_lang:
            # Если язык не определён, используем langdetect как запасной вариант
            detected_lang = detect(text)
            logging.info(f"Язык определён через langdetect: {detected_lang}")

        logging.info(f"Распознанный текст для перевода: {text} (язык: {detected_lang})")
        translation_data[user_id] = {
            "text": text,
            "video_path": video_path,
            "audio_path": audio_path,
            "duration": duration,
            "source_lang": detected_lang
        }
        bot.reply_to(message, f"Распознанный текст: {text}\nЯзык: {detected_lang}\nВыбери пол голоса для перевода или измени язык:", reply_markup=create_recognition_options_keyboard())
    except Exception as e:
        logging.error(f"Ошибка в handle_video_note (translate_audio_video): {e}")
        bot.reply_to(message, "Произошла ошибка при обработке кружочка.", reply_markup=create_main_menu())

# Обработка выбора пола, языка и изменения языка распознавания
def callback_inline(bot, call, create_main_menu):
    try:
        user_id = call.message.chat.id
        data = call.data

        # Обработка перевода голосового сообщения (без выбора пола)
        if data.startswith("translate_"):
            action, source_lang, target_lang = data.split("_")

            if user_id not in pending_actions or pending_actions[user_id]["type"] != "translate_audio_video":
                bot.answer_callback_query(call.id, "🎶 Сначала отправь аудио или видео для перевода.")
                return

            original_text = pending_actions[user_id]["original_text"]
            voice_path = pending_actions[user_id]["voice_path"]
            audio_path = pending_actions[user_id]["audio_path"]

            # Переводим текст
            translated_text = translator.translate(original_text, src=source_lang, dest=target_lang).text
            logging.info(f"Переведённый текст: {translated_text} (с {source_lang} на {target_lang})")

            # Конвертируем переведённый текст в голосовое сообщение
            tts = gTTS(text=translated_text, lang=target_lang)
            tts_file = f"translated_{user_id}.mp3"
            tts.save(tts_file)

            # Конвертируем mp3 в ogg
            audio = AudioSegment.from_mp3(tts_file)
            ogg_file = f"translated_{user_id}.ogg"
            audio.export(ogg_file, format="ogg")

            # Отправляем голосовое сообщение
            with open(ogg_file, "rb") as voice:
                bot.send_voice(user_id, voice, caption=f"📩 Переведённый текст ({target_lang}): {translated_text}", reply_markup=create_main_menu())

            # Удаляем временные файлы с задержкой
            time.sleep(5)  # Задержка 5 секунд
            safe_remove(voice_path)
            safe_remove(audio_path)
            safe_remove(tts_file)
            safe_remove(ogg_file)

            if user_id in pending_actions:
                del pending_actions[user_id]

        # Обработка изменения языка распознавания
        elif data == "change_lang":
            if user_id not in translation_data:
                bot.answer_callback_query(call.id, "Данные для распознавания не найдены. Попробуй снова.")
                return
            bot.edit_message_text(chat_id=call.message.chat.id, message_id=call.message.message_id,
                                  text="Выбери язык для повторного распознавания:", reply_markup=create_recognition_language_keyboard())

        elif data.startswith("recog_lang_"):
            new_lang = data.split("_")[2]
            logging.info(f"Пользователь {user_id} выбрал язык для повторного распознавания: {new_lang}")

            if user_id not in translation_data:
                bot.answer_callback_query(call.id, "Данные для распознавания не найдены. Попробуй снова.")
                return

            audio_path = translation_data[user_id]["audio_path"]
            # Повторно распознаём речь с указанным языком
            result = recognize_speech(audio_path, forced_lang=new_lang)
            text, detected_lang = result if isinstance(result, tuple) else (result, None)
            if "Ошибка" in text or "Не удалось" in text:
                bot.edit_message_text(chat_id=call.message.chat.id, message_id=call.message.message_id,
                                      text=text, reply_markup=create_main_menu())
                return

            # Обновляем данные
            translation_data[user_id]["text"] = text
            translation_data[user_id]["source_lang"] = detected_lang

            bot.edit_message_text(chat_id=call.message.chat.id, message_id=call.message.message_id,
                                  text=f"Распознанный текст: {text}\nЯзык: {detected_lang}\nВыбери пол голоса для перевода или измени язык:",
                                  reply_markup=create_recognition_options_keyboard())

        # Обработка перевода кружочка или голосового сообщения (с выбором пола и языка)
        elif data.startswith("gender_"):
            gender = data.split("_")[1]
            translation_data[user_id]["gender"] = gender
            bot.edit_message_text(chat_id=call.message.chat.id, message_id=call.message.message_id,
                                  text="Теперь выбери язык для перевода:", reply_markup=create_language_keyboard())
        elif data.startswith("lang_"):
            target_lang = data.split("_")[1]
            logging.info(f"Пользователь {user_id} выбрал язык для перевода: {target_lang}")

            if user_id not in translation_data:
                bot.reply_to(call.message, "Данные для перевода не найдены. Попробуй снова.", reply_markup=create_main_menu())
                return

            text = translation_data[user_id]["text"]
            source_lang = translation_data[user_id]["source_lang"]
            logging.info(f"Текст для перевода: {text}")

            # Переводим текст
            try:
                translated_text = translator.translate(text, src=source_lang, dest=target_lang).text
                logging.info(f"Переведённый текст: {translated_text}")
            except Exception as e:
                logging.error(f"Ошибка при переводе текста: {e}")
                bot.reply_to(call.message, "Ошибка при переводе текста.", reply_markup=create_main_menu())
                return

            # Определяем язык для синтеза речи
            tts_lang = target_lang
            if target_lang == "uk":
                tts_lang = "uk"
            elif target_lang == "ru":
                tts_lang = "ru"
            elif target_lang == "en":
                tts_lang = "en"
            logging.info(f"Язык для синтеза речи (gTTS): {tts_lang}")

            # Определяем голос (мужской или женский)
            gender = translation_data[user_id]["gender"]
            tld = "com" if gender == "male" else "co.uk"
            logging.info(f"Выбранный голос: {gender}, tld={tld}")

            # Синтезируем речь
            try:
                tts = gTTS(text=translated_text, lang=tts_lang, tld=tld, slow=False)
                audio_path = f"translated_{user_id}.mp3"
                tts.save(audio_path)
                logging.info(f"Синтезированная речь сохранена: {audio_path}")
            except Exception as e:
                logging.error(f"Ошибка при синтезе речи: {e}")
                bot.reply_to(call.message, "Ошибка при синтезе речи.", reply_markup=create_main_menu())
                return

            # Если это кружочек, создаём новое видео
            if "video_path" in translation_data[user_id]:
                video_path = translation_data[user_id]["video_path"]
                duration = translation_data[user_id]["duration"]
                output_video_path = f"translated_video_{user_id}.mp4"

                try:
                    # Извлекаем видео без звука
                    video_stream = ffmpeg.input(video_path)
                    video_stream = ffmpeg.output(video_stream, "temp_video.mp4", an=None, t=duration, vcodec='copy', f='mp4')
                    ffmpeg.run(video_stream, overwrite_output=True)

                    # Комбинируем видео и новый аудио
                    video = ffmpeg.input("temp_video.mp4")
                    audio = ffmpeg.input(audio_path)
                    output = ffmpeg.output(video, audio, output_video_path, vcodec='copy', acodec='aac', t=duration, f='mp4')
                    ffmpeg.run(output, overwrite_output=True)

                    with open(output_video_path, "rb") as video_file:
                        bot.send_video_note(user_id, video_file, duration=int(duration), reply_markup=create_main_menu())
                    logging.info(f"Переведённый кружочек отправлен пользователю {user_id}")

                    # Удаляем временные файлы
                    for file in [video_path, audio_path, "temp_video.mp4", output_video_path, translation_data[user_id]["audio_path"]]:
                        try:
                            if os.path.exists(file):
                                os.remove(file)
                                logging.info(f"Файл удалён: {file}")
                        except Exception as e:
                            logging.warning(f"Не удалось удалить файл {file}: {e}")
                except Exception as e:
                    logging.error(f"Ошибка при создании переведённого кружочка: {e}")
                    bot.reply_to(call.message, "Ошибка при создании переведённого кружочка.", reply_markup=create_main_menu())
                    return
            else:
                # Если это голосовое сообщение, отправляем аудио
                with open(audio_path, "rb") as audio_file:
                    bot.send_audio(user_id, audio_file, reply_markup=create_main_menu())
                logging.info(f"Переведённое аудио отправлено пользователю {user_id}")

                # Удаляем временные файлы
                for file in [translation_data[user_id]["voice_path"], translation_data[user_id]["audio_path"], audio_path]:
                    try:
                        if os.path.exists(file):
                            os.remove(file)
                            logging.info(f"Файл удалён: {file}")
                    except Exception as e:
                        logging.warning(f"Не удалось удалить файл {file}: {e}")

            if user_id in translation_data:
                del translation_data[user_id]

    except Exception as e:
        logging.error(f"Ошибка в callback_inline (translate_audio_video): {e}")
        bot.reply_to(call.message, "Произошла ошибка при обработке перевода.", reply_markup=create_main_menu())