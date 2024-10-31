# usage: source this file, then `cat yourLog.txt | log2jsonl > yourLog.jsonl` and `pd.read_jsons('yourLog.jsonl', lines=True)`

log2jsonl() {
    grep tracing | sed 's/.*tracing:[0-9]*://' | sed 's/at=\(.*\)/"at": \1/' | sed 's/\([^=]*\)=\([^;]*\);/"\1": "\2", /g' | sed 's/.*/{&}/'
}


