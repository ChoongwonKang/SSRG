import csv
import asyncio
import aiohttp
import os
import logging
from tqdm.asyncio import tqdm_asyncio

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

O1_API_KEY = os.getenv("O1_API_KEY", "")
O1_API_ENDPOINT = ""  

async def generate_sarcasm_reasoning(session, scene_id, context_rows, target_row, index):

    # unpack target info
    speaker = target_row["speaker"]
    fau_result = target_row["FAU_result"]
    sentence = target_row["sentence"]
    audio_info = target_row["Audio"]
    sarcasm_type = target_row["sarcasm_type"]

    # Build a "context" string from all _c_?? rows
    # We'll include speaker, sentence, FAU_result, Audio, etc.
    # and label them in a chronological order if needed.
    context_str = ""
    for i, c_row in enumerate(context_rows):
        c_key = c_row["key"]
        c_speaker = c_row["speaker"]
        c_fau = c_row["FAU_result"]
        c_sentence = c_row["sentence"]
        c_audio = c_row["Audio"]
        context_str += (
            f"  \nSpeaker: {c_speaker}\n"
            f"  FAU: {c_fau}\n"
            f"  Audio [F0 mean, F0 var, Energy mean, Energy var, Jitter, Shimmer]: {c_audio}\n"
            f"  Utterance: {c_sentence}\n\n"
        )

    # Now we incorporate context + target into one big system_instructions
    system_instructions = f"""
Task:
You are a helpful assistant trained to inference a sentence into one of three sarcasm types: propositional, embedded, or illocutionary.

We have three sarcasm types:
1) Propositional: Opposite of actual meaning
2) Embedded: Contradictory word/phrase in context
3) Illocutionary: Tone/cues that oppose literal meaning
"""

    user_prompt = f"""
Explain why the target (final line) is sarcastic, using the provided context (previous lines) and multimodal cues (FAU, Audio).
Each answer must start with "This is {sarcasm_type} sarcasm," followed by "since" or "because" (no extra periods).
Keep them under 25 words if possible.
Context (previous lines): {context_str}

Target (final line):
Speaker: {speaker}
 FAU: {fau_result}
 Audio (F0 mean, F0 var, Energy mean, Energy var, Jitter, Shimmer): {audio_info}
 Utterance: {sentence}

Sarcasm type: {sarcasm_type}

Why does the final line belong to {sarcasm_type} sarcasm? 
Please provide answers in the required format.

"""
    #print(system_instructions)
    #print(user_prompt)
    # Build ChatCompletion-like payload
    data = {
    "model": "gpt-4o",
    "messages": [
        {"role": "system", "content": system_instructions.strip()},
        {"role": "user", "content": user_prompt.strip()}
    ],
    "max_completion_tokens": 2048,
    "temperature": 0.3
    }

    url = ""  
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {O1_API_KEY}"
    }

    retries = 3
    timeout = 300

    for attempt in range(1, retries + 1):
        try:
            async with session.post(url, headers=headers, json=data, timeout=timeout) as response:
                result = await response.json()

                logger.debug(f"Raw response for scene {scene_id} (index {index}): {result}")

                if 'choices' not in result:
                    logger.warning(f"Scene {scene_id} - Unexpected response format: {result}")
                    return None

                return result['choices'][0]['message']['content']
        except Exception as e:
            logger.error(f"Scene {scene_id} - Error on attempt {attempt}/{retries}: {e}")
            if attempt == retries:
                raise
            else:
                wait_time = attempt * 2
                logger.info(f"Retrying scene {scene_id} in {wait_time} seconds...")
                await asyncio.sleep(wait_time)

    return None

async def main():
    csv_file_path = r""
    output_file_path = r""

    async with aiohttp.ClientSession() as session:
        with open(csv_file_path, mode="r", encoding="utf-8-sig") as csvfile:
            reader = csv.DictReader(csvfile)
            all_rows = list(reader)

        sarcasm_true_rows = [row for row in all_rows if str(row.get("sarcasm", "")).strip().lower() == "true"]
        logger.info(f"[1] sarcasm==true : {len(sarcasm_true_rows)}")

        from collections import defaultdict
        scene_map = defaultdict(lambda: {"context": [], "target": None})

        for row in all_rows:
            key_val = row["key"]

            if "_u" in key_val:
                scene_id = key_val.replace("_u", "")
            elif "_c" in key_val:
                scene_id = key_val.split("_c")[0]
            else:
                scene_id = key_val  # fallback

            if key_val.endswith("_u"):
                scene_map[scene_id]["target"] = row
            else:
                scene_map[scene_id]["context"].append(row)

        tasks = []
        scene_ids = []

        skipped_no_target = []
        skipped_false = []
        skipped_like = []

        for scene_id, data_dict in scene_map.items():
            target = data_dict["target"]
            context = data_dict["context"]

            if not target:
                skipped_no_target.append(scene_id)
                continue

            sarcasm_type = str(target.get("sarcasm_type", "")).strip().lower()
            sarcasm_flag = str(target.get("sarcasm", "")).strip().lower()

            if sarcasm_flag == "false":
                skipped_false.append(scene_id)
                continue
            if sarcasm_type.startswith("like-"):
                skipped_like.append(scene_id)
                continue

            tasks.append(
                asyncio.create_task(generate_sarcasm_reasoning(session, scene_id, context, target, len(tasks)))
            )
            scene_ids.append(scene_id)

        logger.info(f"[2]scene: {len(scene_ids)}")
        logger.info(f"[3] No _u target scene : {len(skipped_no_target)}")
        logger.info(f"[4] Sarcasm == false scene : {len(skipped_false)}")
        logger.info(f"[5] Sarcasm_type == like-* scene: {len(skipped_like)}")

        logger.info(f"→ {len(tasks)}")
        results = await tqdm_asyncio.gather(*tasks, desc="Processing Scenes", total=len(tasks))

        out_rows = []
        for i, scene_id in enumerate(scene_ids):
            reasoning_text = results[i]
            if not reasoning_text or isinstance(reasoning_text, Exception):
                reasoning_text = ""
            target_key = scene_map[scene_id]["target"]["key"]
            out_rows.append({
                "scene_id": scene_id,
                "key": target_key,
                "resoning": reasoning_text
            })

        fieldnames = ["scene_id", "key", "resoning"]
        with open(output_file_path, "w", encoding="utf-8-sig", newline="") as outfile:
            writer = csv.DictWriter(outfile, fieldnames=fieldnames)
            writer.writeheader()
            for row_dict in out_rows:
                writer.writerow(row_dict)

        logger.info(f"[✔] save results: {output_file_path}")



if __name__ == "__main__":
    asyncio.run(main())
