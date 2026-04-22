DATA_DIR="data/wikidata_dolma/"
MATCH_WORD="wiki"
PARALLEL_DOWNLOADS="2"
DOLMA_VERSION="v1_6"

git clone https://huggingface.co/datasets/allenai/dolma
mkdir -p "${DATA_DIR}"


cat "dolma/urls/${DOLMA_VERSION}.txt" | grep "${MATCH_WORD}" | xargs -n 1 -P "${PARALLEL_DOWNLOADS}" wget -q -P "$DATA_DIR"
