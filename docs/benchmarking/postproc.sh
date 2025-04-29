# usage: source this file, then `cat yourLog.txt | log2jsonl > yourLog.jsonl` and `pd.read_jsons('yourLog.jsonl', lines=True)`
# you may want to split the jsonl file in three because of the different schemata, tho: `log2jsnol yourLog.txt` does exactly that
# NOTE dont use, replaced by cascade.benchmarks.reporting . Preseved for making the notebooks here reproducible

log2jsonl() {
    grep tracing | sed 's/.*tracing:[0-9]*://' | sed 's/at=\(.*\)/"at": \1/' | sed 's/\([^=]*\)=\([^;]*\);/"\1": "\2", /g' | sed 's/.*/{&}/'
}

log2jsonlSplit() {
    F=$1
    O1="${F%.*}".tasks.jsonl
    O2="${F%.*}".datasets.jsonl
    O3="${F%.*}".controller.jsonl
    cat $F | grep 'action=task' | log2jsonl > "$O1"
    cat $F | grep 'action=transmit' | log2jsonl > "$O2"
    cat $F | grep 'action=controller' | log2jsonl > "$O3"
}
