import ffmpeg
import os
import logging

# Настройка логирования
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Функция конвертации видео в кружочек (с сохранением звука)
def convert_to_circle(video_path, output_path="circle_video.mp4"):
    logging.info(f"Конвертация видео в кружочек: {video_path}")
    try:
        # Получаем информацию о видео
        probe = ffmpeg.probe(video_path)
        video_info = next(s for s in probe['streams'] if s['codec_type'] == 'video')
        width = int(video_info['width'])
        height = int(video_info['height'])
        duration = float(probe['format']['duration'])

        # Проверяем наличие аудиопотока
        has_audio = any(stream['codec_type'] == 'audio' for stream in probe['streams'])
        logging.info(f"Видео имеет аудиопоток: {has_audio}")

        # Ограничиваем длительность до 60 секунд (максимум для кружочков в Telegram)
        if duration > 60:
            duration = 60

        # Определяем размер для квадратного видео
        size = min(width, height)
        crop_filter = f"crop={size}:{size}:{(width - size)//2}:{(height - size)//2}"

        # Конвертируем видео с сохранением аудио
        video_stream = ffmpeg.input(video_path, t=duration)
        video_stream = ffmpeg.filter(video_stream, 'crop', size, size, (width - size) // 2, (height - size) // 2)
        video_stream = ffmpeg.filter(video_stream, 'scale', 512, 512, force_original_aspect_ratio='decrease')
        video_stream = ffmpeg.filter(video_stream, 'pad', 512, 512, '(ow-iw)/2', '(oh-ih)/2')
        video_stream = ffmpeg.filter(video_stream, 'setsar', 1)

        # Если есть аудиопоток, добавляем его
        if has_audio:
            audio_stream = ffmpeg.input(video_path, t=duration)
            output = ffmpeg.output(
                video_stream, audio_stream, output_path,
                vcodec='libx264',  # Используем libx264 для H.264
                acodec='aac',      # Кодек аудио AAC
                pix_fmt='yuv420p', # Формат пикселей, который Telegram принимает
                r=30,              # Частота кадров
                profile='baseline',# Профиль H.264 для совместимости
                level=3.0,         # Уровень H.264
                t=duration,
                video_bitrate='1M',# Ограничиваем битрейт видео
                audio_bitrate='128k', # Битрейт аудио
                **{'movflags': 'faststart', 'f': 'mp4'}  # Формат MP4 и оптимизация
            )
        else:
            # Если аудио нет, создаём видео без аудиопотока
            output = ffmpeg.output(
                video_stream, output_path,
                vcodec='libx264',
                pix_fmt='yuv420p',
                r=30,
                profile='baseline',
                level=3.0,
                t=duration,
                video_bitrate='1M',
                **{'movflags': 'faststart', 'f': 'mp4'}
            )

        ffmpeg.run(output, overwrite_output=True)
        logging.info(f"Кружочек создан: {output_path}")

        # Проверяем, есть ли аудио в итоговом файле
        probe_output = ffmpeg.probe(output_path)
        has_audio_output = any(stream['codec_type'] == 'audio' for stream in probe_output['streams'])
        logging.info(f"Итоговый файл имеет аудиопоток: {has_audio_output}")

        return output_path
    except ffmpeg.Error as e:
        logging.error(f"Ошибка при конвертации видео в кружочек: {e}")
        return None