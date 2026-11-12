# The local QLever endpoint

All queries of the test run were executed on a QLever instance on a laptop, with an index built from an official Wikidata dump. This note records what was used, so that the run can be repeated or compared with another setup.

## Setup

| | |
|---|---|
| Machine | laptop, AMD Ryzen AI 9, 32 GB RAM, one 1 TB SSD; Windows 11 with WSL 2 (Ubuntu 24.04, memory limit 24 GB) |
| QLever | 0.6.0 (git 6c5861a), from the official apt repository |
| Input | `wikidata-20261105-all-BETA.ttl.bz2` (126 GB, SHA-1 `7a3fb9926da6b19f7ac63d862526e2286dbcfb51`) and `wikidata-20261106-lexemes-BETA.ttl.bz2` (0.7 GB, SHA-1 `f6fb87107d5d4fce74691254dfc315ee29cf561b`) from `https://dumps.wikimedia.org/wikidatawiki/entities/` |
| Index configuration | the Qleverfile that `qlever setup-config wikidata` writes, with the two dated files above as input instead of `latest`, and without `dcatap.nt` (metadata about the dump itself) |
| Index | 21,663,690,797 triples; 507 GB (472 GiB) on disk; built in 8 h 10 min; the index folder peaked at about 580 GiB during the build |
| Server | `qlever-server -i wikidata -j 8 -p 7001 -m 14G -c 3G -e 1G -k 200 -s 30s` |

The server options are those of `qlever start` with the memory limits reduced to fit 24 GB (the official configuration uses 20G / 15G / 5G for `-m` / `-c` / `-e`). The 30 s timeout equals the limit of the public endpoint for requests without an access token.

## Steps

```bash
# 1. QLever and its control script: https://github.com/ad-freiburg/qlever (apt package or Docker)
qlever setup-config wikidata          # writes the Qleverfile; edit GET_DATA_CMD / INPUT_FILES to the dated dump files
# 2. download the two dump files and check them against the published SHA-1 values
# 3. build (see the note on disk space below; 24 GB RAM were enough)
qlever index
# 4. start
qlever start                          # or the qlever-server command above
# 5. run the queries
python scripts/run_queries.py --endpoint http://127.0.0.1:7001
```

## Things worth knowing

- **Disk.** The build needs more than the final index: temporary vocabularies and unsorted triples exist alongside the final files for a while (580 GiB at the peak, against 472 GiB at the end), and the compressed dump (118 GiB) has to be there until it has been read. On a single 1 TB disk this only fits if the dump is deleted once it has been read, which is what I did. Under WSL the virtual disk never shrinks on its own, so a failed attempt means recreating the WSL distribution.
- **Wikimedia keeps a dump for a few weeks only.** Rebuilding the same index later requires a copy of the two files above. The released execution results do not depend on the index: `scripts/evaluate.py` runs without it.
- **Differences from the public endpoint.** `https://qlever.dev/api/wikidata` serves a preprocessed dump with live updates and a development version of QLever. On the development questions the two gave the same verdict for all gold queries and the same execution status for 3,232 of 3,245 predictions.
- **Server responses the executor has to handle** (`src/kgqa/execute.py`): `send=N` limits the rows sent back while the full count is still reported; an error after the first rows have been streamed arrives as HTTP 200 with a note appended to broken JSON; a query cancelled because QLever estimates that it cannot finish in time arrives as HTTP 429 without the word "timeout".
- After the test run the index and the WSL distribution were deleted; the build log, the configuration and the server log were kept.
