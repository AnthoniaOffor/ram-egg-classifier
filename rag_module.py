rag_module_code = r'''
import re
import torch
from sentence_transformers import SentenceTransformer, util

# ----------------------------
# Knowledge base documents
# ----------------------------
documents = [

    # --- Project overview ---
    "The model supports 21 bird species: Agelaius phoeniceus (Red-winged Blackbird), Ammodramus savannarum (Grasshopper Sparrow), Anthus spinoletta (Water Pipit), Cardellina pusilla (Wilson's Warbler), Certhia americana (Brown Creeper), Cistothorus palustris (Marsh Wren), Euphagus cyanocephalus (Brewer's Blackbird), Geothlypis tolmiei (MacGillivray's Warbler), Geothlypis trichas (Common Yellowthroat), Icteria virens (Yellow-breasted Chat), Junco hyemalis (Dark-eyed Junco), Molothrus ater (Brown-headed Cowbird), Passer domesticus (House Sparrow), Passerella iliaca (Fox Sparrow), Poecile atricapillus (Black-capped Chickadee), Pooecetes gramineus (Vesper Sparrow), Quiscalus quiscula (Common Grackle), Setophaga petechia (Yellow Warbler), Setophaga ruticilla (American Redstart), Vireo olivaceus (Red-eyed Vireo), and Zonotrichia albicollis (White-throated Sparrow)."

    "The IBIS app is designed to help researchers and curators at the Royal Alberta Museum classify bird eggs using images. The system analyzes egg photos and predicts the most likely species among a predefined set of 21 species.",

    "The classification is performed using a YOLO-based computer vision model trained on thousands of egg images. The model learns visual patterns such as egg shape, base color, and marking patterns to distinguish between species.",

    "The system is intended as a support tool, not a replacement for expert judgment. Researchers are expected to validate model predictions by comparing results with known museum specimens, collection records, or reference descriptions.",

    "The model can only classify eggs among the 21 species included in the training dataset. It will always return one of these 21 species, even if the uploaded egg does not actually belong to any of them.",

    "All 21 species in this project were selected because their eggs are relatively small, visually similar, and difficult to distinguish manually. Many of them overlap in size, color, and spotting pattern, and several species also show noticeable variation within the same species.",

    "IBIS is most useful when a researcher already suspects that an egg belongs to one of the 21 supported species, but needs help narrowing the identification. The model prediction can then be checked against collection material or species descriptions.",

    "Egg classification works best when images are taken under conditions similar to the training data, such as clear indoor lighting, minimal shadows, and a simple background. Major differences in lighting, blur, or angle can reduce reliability.",

    "Some species are learned better than others. Performance depends on how much training data was available, how visually consistent the eggs were, and how separable each species was from the others.",

    "The model can improve over time if it is trained with more images and eventually expanded to include additional species.",

    "The chatbot provides short project-aware reference information about the supported species, the appearance of their eggs, and the practical limitations of the model.",

    "In this project, egg descriptions are meant to support visual comparison rather than to provide a full biological account. Color, spotting, and approximate size are especially useful because the app classifies from egg photographs only.",

    # --- Species documents ---
    "Species: Agelaius phoeniceus. Common name: Red-winged Blackbird. This is a common wetland blackbird, especially recognizable in males by the red and yellow shoulder patches. Its eggs are medium-small, oval, about 2.2-2.7 cm long and 1.6-1.9 cm wide, usually pale blue-green to gray with black or brown markings.",

    "Species: Ammodramus savannarum. Common name: Grasshopper Sparrow. This is a small grassland sparrow with buffy plumage and a thin insect-like song. Its eggs are small, oval, about 1.5-2.1 cm long and 1.4-1.5 cm wide, white with light reddish-brown speckles.",

    "Species: Anthus spinoletta. Common name: Water Pipit. This is a slender ground-feeding pipit associated with open country and wet or alpine habitats. Its eggs are small and oval, typically pale to whitish with darker spotting, and pipit eggs often show dense markings toward one end.",

    "Species: Cardellina pusilla. Common name: Wilson's Warbler. This is a very small, active yellow warbler, with males showing a black cap. Its eggs are small, oval, about 1.5-1.8 cm long and 1.2-1.3 cm wide, white to creamy white with fine reddish-brown speckling or spotting.",

    "Species: Certhia americana. Common name: Brown Creeper. This is a tiny bark-colored bird that climbs tree trunks in a creeping spiral. Its eggs are small, oval, about 1.5-1.6 cm long and about 1.2 cm wide, smooth white with pink or reddish-brown speckling.",

    "Species: Cistothorus palustris. Common name: Marsh Wren. This is a tiny marsh bird that lives among cattails and reeds and is known for its energetic song. Its eggs are small, oval, about 1.4-1.8 cm long and 1.1-1.4 cm wide, brown with darker spots.",

    "Species: Euphagus cyanocephalus. Common name: Brewer's Blackbird. This is a medium-sized blackbird of open habitats; males are glossy black and females are plain brown. Its eggs are larger than many of the other species in this project, about 2.3-2.9 cm long and 1.7-2.0 cm wide, pale gray to greenish white and often clouded or spotted with brown, pink, yellow, violet, or gray.",

    "Species: Geothlypis tolmiei. Common name: MacGillivray's Warbler. This is a furtive warbler of dense brush and thickets, with males showing a gray hood and white eye crescents. Its eggs are small, creamy white, and variably tinted or speckled; in practice they fit the same general small, oval, lightly marked pattern seen in several of the warbler species in this project.",

    "Species: Geothlypis trichas. Common name: Common Yellowthroat. This is a common marsh and brushland warbler, with males showing a black facial mask. Its eggs are small, oval, about 1.5-2.0 cm long and 1.2-1.5 cm wide, white with markings that may be gray, lilac, reddish-brown, or black.",

    "Species: Icteria virens. Common name: Yellow-breasted Chat. This is a larger, secretive songbird of dense shrubs, notable for its bright yellow breast and varied vocalizations. Its eggs are medium-small, oval, about 1.8-2.5 cm long and 1.5-1.9 cm wide, white or off-white with red, brown, gray, or purple speckles.",

    "Species: Junco hyemalis. Common name: Dark-eyed Junco. This is a common ground-feeding sparrow with a compact shape and a pale bill. Its eggs are small to medium-small, oval, about 1.9-2.1 cm long and 1.5-1.6 cm wide, white, gray, pale bluish white, or pale greenish white, usually speckled with brown, gray, or green.",

    "Species: Molothrus ater. Common name: Brown-headed Cowbird. This is a stocky blackbird known for brood parasitism, laying its eggs in the nests of other birds. Its eggs are medium-small, oval, about 1.8-2.5 cm long and 1.5-1.8 cm wide, white to grayish white with brown or gray spots.",

    "Species: Passer domesticus. Common name: House Sparrow. This is a familiar human-associated sparrow, especially common around buildings and urban areas. Its eggs are medium-small, oval, about 2.0-2.2 cm long and 1.4-1.6 cm wide, light white to greenish white or bluish white, usually spotted with gray or brown.",

    "Species: Passerella iliaca. Common name: Fox Sparrow. This is a large, strongly marked sparrow, often reddish or rusty in tone. Its eggs are medium-small, oval, about 2.1-2.4 cm long and 1.6-1.8 cm wide, pale bluish green with bold reddish-brown blotches or cloudy markings.",

    "Species: Poecile atricapillus. Common name: Black-capped Chickadee. This is a small woodland bird with a black cap, white cheeks, and active flocking behavior. Its eggs are small, oval, about 1.5 cm long and 1.2 cm wide, white with fine reddish-brown dots or spots.",

    "Species: Pooecetes gramineus. Common name: Vesper Sparrow. This is a grassland sparrow with a relatively plain face and white outer tail feathers visible in flight. Its eggs are medium-small, oval, about 1.9-2.3 cm long and 1.3-1.7 cm wide, whitish with variable brown or purplish spots, streaks, and blotches.",

    "Species: Quiscalus quiscula. Common name: Common Grackle. This is a large, long-tailed blackbird with glossy iridescent plumage. Its eggs are among the larger ones in this set, oval, about 2.5-3.3 cm long and 1.9-2.3 cm wide, and can range from light blue and pearl gray to white or dark brown, usually with brown spotting.",

    "Species: Setophaga petechia. Common name: Yellow Warbler. This is a bright yellow warbler often found in shrubby or wet habitats. Its eggs are small, oval, about 1.5-2.1 cm long and 1.2-1.6 cm wide, grayish or greenish white with dark spots.",

    "Species: Setophaga ruticilla. Common name: American Redstart. This is an active forest warbler that flashes bright patches in the wings and tail while foraging. Its eggs are small, white to creamy, with brownish or reddish blotches; some can be so heavily speckled that they appear almost brown overall.",

    "Species: Vireo olivaceus. Common name: Red-eyed Vireo. This is a common summer forest bird, especially known for its persistent singing. Its eggs are small to medium-small, oval, about 2.0-2.4 cm long, dull white, and marked with sparse sepia speckling.",

    "Species: Zonotrichia albicollis. Common name: White-throated Sparrow. This is a recognizable sparrow with a striped head and a clean, crisp facial pattern. Its eggs are medium-small, oval, about 1.9-2.3 cm long and 1.4-1.7 cm wide, very pale blue or greenish blue, speckled with purplish, chestnut, and lilac markings."
]

# ----------------------------
# Species lookup
# ----------------------------
species_names = [
    "agelaius phoeniceus",
    "ammodramus savannarum",
    "anthus spinoletta",
    "cardellina pusilla",
    "certhia americana",
    "cistothorus palustris",
    "euphagus cyanocephalus",
    "geothlypis tolmiei",
    "geothlypis trichas",
    "icteria virens",
    "junco hyemalis",
    "molothrus ater",
    "passer domesticus",
    "passerella iliaca",
    "poecile atricapillus",
    "pooecetes gramineus",
    "quiscalus quiscula",
    "setophaga petechia",
    "setophaga ruticilla",
    "vireo olivaceus",
    "zonotrichia albicollis",
]

species_doc_map = {}
for doc in documents:
    lower_doc = doc.lower()
    if lower_doc.startswith("species:"):
        for sp in species_names:
            if f"species: {sp}." in lower_doc:
                species_doc_map[sp] = doc
                break

# ----------------------------
# Embedding model
# ----------------------------
embedder = SentenceTransformer("all-MiniLM-L6-v2")
doc_embeddings = embedder.encode(documents, convert_to_tensor=True)

def normalize_text(text):
    return re.sub(r"[_\s]+", " ", text.lower()).strip()

def find_species_in_query(query):
    q = normalize_text(query)
    for sp in species_names:
        if sp in q:
            return sp
    return None

# ----------------------------
# Retrieval
# ----------------------------
def retrieve_context(query, top_k=3):
    matched_species = find_species_in_query(query)

    # If a species name is explicitly in the query, force that document first
    if matched_species is not None:
        selected_docs = [species_doc_map[matched_species]]

        # Add a couple of general docs for extra context
        general_docs = [
            "The chatbot provides general information about the species, their eggs, and the project, based only on the available dataset and predefined descriptions.",
            "All species included in this project were selected because their eggs are visually similar and difficult to distinguish, even for human experts. This makes the classification task more challenging and realistic."
        ]

        for doc in general_docs:
            if len(selected_docs) < top_k:
                selected_docs.append(doc)

        selected_scores = [1.0] + [0.5] * (len(selected_docs) - 1)
        return selected_docs, selected_scores

    # Otherwise use semantic retrieval
    query_embedding = embedder.encode(query, convert_to_tensor=True)
    scores = util.cos_sim(query_embedding, doc_embeddings)[0]
    k = min(top_k, len(documents))
    top_results = torch.topk(scores, k=k)

    selected_docs = [documents[idx] for idx in top_results.indices.tolist()]
    selected_scores = [float(scores[idx]) for idx in top_results.indices.tolist()]
    return selected_docs, selected_scores

# ----------------------------
# Answer generation
# ----------------------------
def clean_text(text):
    return re.sub(r"\s+", " ", text).strip()

def answer_from_context(query, retrieved_docs):
    query_lower = query.lower()
    matched_species = find_species_in_query(query)

    if matched_species is not None:
        return clean_text(retrieved_docs[0])

    if any(term in query_lower for term in ["what does the app do", "purpose", "project", "limitation", "limitations", "researcher", "museum", "model"]):
        return clean_text(" ".join(retrieved_docs[:2]))

    return clean_text(" ".join(retrieved_docs[:2]))

def rag_chatbot(query):
    if query is None or not str(query).strip():
        return "Please enter a question about the project, the supported species, or the eggs."

    retrieved_docs, scores = retrieve_context(query, top_k=3)
    return answer_from_context(query, retrieved_docs)

def rag_chatbot_with_sources(query):
    if query is None or not str(query).strip():
        return (
            "Please enter a question about the project, the supported species, or the eggs.",
            "No context retrieved."
        )

    retrieved_docs, scores = retrieve_context(query, top_k=3)
    answer = answer_from_context(query, retrieved_docs)

    context_text = ""
    for i, (doc, score) in enumerate(zip(retrieved_docs, scores), start=1):
        context_text += f"{i}. (score={score:.3f}) {doc}\n\n"

    return answer, context_text
'''

with open("rag_module.py", "w", encoding="utf-8") as f:
    f.write(rag_module_code)

print("rag_module.py saved successfully.")
