import whisperx
import subprocess

device = "cuda" 
batch_size = 16 # reduce if low on GPU mem
compute_type = "float16" # change to "int8" if low on GPU mem (may reduce accuracy)

model = whisperx.load_model("large-v2", device, compute_type=compute_type, language="en")

hugging_face_api_key = "your-api-key"

conversation_1 = [
    "conversation_1_part_1",
    "conversation_1_part_2",
    "conversation_1_part_3",
    "conversation_1_part_4"
]

conversation_2 = [
    "conversation_2_part_1",
    "conversation_2_part_2",
    "conversation_2_part_3",
    "conversation_2_part_4",
]

conversation_3 = [
    "conversation_3_part_1",
    "conversation_3_part_2",
    "conversation_3_part_3",
    "conversation_3_part_4",
]

conversations = [conversation_1, conversation_2, conversation_3]

def concatenate_audios_ffmpeg(file_list, output_filename):
    # Prepare the list file
    with open("audio_list.txt", "w") as file:
        for audio_file in file_list:
            file.write(f"file '{audio_file}'\n")
    
    # Run the FFmpeg command
    command = [
        "ffmpeg",
        "-f", "concat",
        "-safe", "0",
        "-i", "audio_list.txt",
        "-c", "copy",
        output_filename
    ]
    subprocess.run(command, check=True)
    return whisperx.load_audio(output_filename)

# 1. Do the work
for audio_list in conversations:
    list_filename = "audio_list.txt"
    combined_audio_name = f"combined_audio_{audio_list[0][:13]}.mp3"

    audio = concatenate_audios_ffmpeg(audio_list, combined_audio_name)
    print("audio concatenated")

    result = model.transcribe(audio, batch_size=batch_size, language="en")

    # 2. Align whisper output
    model_a, metadata = whisperx.load_align_model(language_code="en", device=device)
    result = whisperx.align(result["segments"], model_a, metadata, audio, device, return_char_alignments=False)

    # 3. Assign speaker labels
    diarize_model = whisperx.DiarizationPipeline(model_name="pyannote/speaker-diarization-3.1", use_auth_token=hugging_face_api_key, device=device)
    diarize_segments = diarize_model(audio)
    result = whisperx.assign_word_speakers(diarize_segments, result)

    # Convert segments to full sentences with speaker attribution
    transcription = []
    current_speaker = None
    current_sentence = []

    for segment in result["segments"]:
        speaker = segment.get("speaker", "Unknown")
        text = segment["text"]

        if speaker != current_speaker:
            if current_sentence:
                transcription.append(f"Speaker {current_speaker}: {' '.join(current_sentence)}")
                current_sentence = []
            current_speaker = speaker

        current_sentence.append(text)

    if current_sentence:
        transcription.append(f"Speaker {current_speaker}: {' '.join(current_sentence)}")

    # Output the formatted transcription to a file
    with open(f"transcription_output_{audio_list[0][:4]}.txt", "w", encoding="utf-8") as file:
        for line in transcription:
            file.write(line + "\n")