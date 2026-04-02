import anthropic
import json
import time

CATEGORY_PROMPT = """You are classifying NYT Connections puzzle groups into categories following the taxonomy of Samdarshi et al. (2024).

The four categories are:
1. SEMANTIC — Groups based on synonymy, hypernymy, or direct semantic relations. Examples: "DAYS OF THE WEEK", "METAL ELEMENTS", "TYPES OF DOGS"
2. ENCYCLOPEDIC — Groups requiring cultural or world knowledge. Examples: "W.N.B.A. TEAMS", "OSCAR BEST PICTURE WINNERS", "CAPITALS OF EUROPE"
3. WORDPLAY — Groups based on word form, morphology, phonology, or multiword expressions. Examples: "___LAND", "B-___", "WORDS THAT RHYME WITH CAT", "ANAGRAMS OF ___"
4. ASSOCIATIVE — Groups based on thematic or loose associative connections that are neither purely semantic nor encyclopedic. Examples: "THINGS WITH TRUNKS", "SPHERICAL FOODS", "THINGS IN A KITCHEN"

Given a group description and its members, classify it into exactly one of these four categories.
Respond with ONLY a JSON object in this format:
{"category": "SEMANTIC", "reasoning": "brief explanation"}

The category must be one of: SEMANTIC, ENCYCLOPEDIC, WORDPLAY, ASSOCIATIVE"""


def classify_group(description, members, client=None):
    """
    Classify a single group description using Claude.
    
    Args:
        description: group description string e.g. "DAYS OF THE WEEK"
        members: list of 4 words in the group
        client: anthropic client (created if not provided)
    
    Returns:
        dict with category and reasoning
    """
    if client is None:
        client = anthropic.Anthropic()

    prompt = f"Group description: {description}\nMembers: {', '.join(members)}"

    message = client.messages.create(
        model="claude-sonnet-4-20250514",
        max_tokens=200,
        system=CATEGORY_PROMPT,
        messages=[{"role": "user", "content": prompt}]
    )

    response_text = message.content[0].text.strip()
    
    try:
        result = json.loads(response_text)
        return result
    except json.JSONDecodeError:
        # fallback if JSON parsing fails
        return {"category": "ASSOCIATIVE", "reasoning": "parse error"}


def classify_all_puzzles(puzzles, delay=0.05):
    """
    Classify all groups across all puzzles using parallel API calls.
    """
    from concurrent.futures import ThreadPoolExecutor, as_completed
    import time
    
    client = anthropic.Anthropic()
    classifications = {}
    
    all_groups = []
    for puzzle in puzzles:
        for group in puzzle["groups"]:
            all_groups.append({
                "puzzle_id": puzzle["puzzle_id"],
                "level": group["level"],
                "description": group["group"],
                "members": group["members"]
            })
    
    total = len(all_groups)
    done = 0
    
    def classify_one(item):
        max_retries = 3
        for attempt in range(max_retries):
            try:
                result = classify_group(item["description"], item["members"], client)
                return {
                    "puzzle_id": item["puzzle_id"],
                    "level": item["level"],
                    "description": item["description"],
                    "members": item["members"],
                    "category": result.get("category", "ASSOCIATIVE"),
                    "reasoning": result.get("reasoning", "")
                }
            except Exception as e:
                if "rate_limit" in str(e).lower() and attempt < max_retries - 1:
                    time.sleep(5 * (attempt + 1))
                else:
                    return {
                        "puzzle_id": item["puzzle_id"],
                        "level": item["level"],
                        "description": item["description"],
                        "members": item["members"],
                        "category": "ASSOCIATIVE",
                        "reasoning": f"error: {str(e)}"
                    }
    
    with ThreadPoolExecutor(max_workers=5) as executor:
        futures = {executor.submit(classify_one, g): g for g in all_groups}
        for future in as_completed(futures):
            result = future.result()
            key = str((result["puzzle_id"], result["level"]))
            classifications[key] = result
            done += 1
            if done % 50 == 0:
                print(f"  {done}/{total} groups classified")
    
    return classifications