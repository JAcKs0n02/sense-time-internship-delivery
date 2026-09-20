# Week 2 Day 6: Data Collection and Formatting

Day 6 collected three fixed-revision Chinese instruction datasets and selected
2,000 Alpaca-GPT4-zh records, 2,000 COIG-PC records and 1,000 ShareGPT-zh
records. The same 5,000 semantic samples were converted into standard Alpaca
and ShareGPT representations.

## Submitted Files

| Requirement | Submitted file |
|---|---|
| Standard Alpaca format (`instruction`, `input`, `output`) | [Formatted_Alpaca_5K.jsonl](Data/Formatted_Alpaca_5K.jsonl) |
| Standard ShareGPT format (`conversations`) | [Formatted_ShareGPT_5K.jsonl](Data/Formatted_ShareGPT_5K.jsonl) |
| Stable sample identity and source traceability | [Data_Provenance.jsonl](Data/Data_Provenance.jsonl) |
| Human-readable raw-data archive manifest | [Raw_Data_Manifest.md](Manifest/Raw_Data_Manifest.md) |
| Machine-readable raw-data archive manifest | [Raw_Data_Manifest.csv](Manifest/Raw_Data_Manifest.csv) |
| Fixed revisions and collection metadata | [Collection_Metadata.json](Manifest/Collection_Metadata.json) |

The Alpaca and ShareGPT files are two equivalent representations of the same
5,000 samples. They must not be added together and treated as a 10,000-record
training set.

## Verified Counts

| Dataset | Records |
|---|---:|
| Alpaca-GPT4-zh subset | 2,000 |
| COIG-PC subset | 2,000 |
| ShareGPT-zh subset | 1,000 |
| Formatted Alpaca data | 5,000 |
| Formatted ShareGPT data | 5,000 |
| Provenance records | 5,000 |

The complete raw subsets, collection/conversion scripts, tests and execution
logs remain in the repository engineering archive at
`deliverables/week2/day6/`. They are intentionally omitted here because the
teacher-facing deliverable for Day 6 is the formatted data and original-data
archive manifest.
