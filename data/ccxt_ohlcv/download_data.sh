#!/bin/sh

START_DATE="2021-11-01"
END_DATE="2022-01-17"

SYMBOLS=(
    "BTC" "ETH"
    # Uncomment this to use the "ccxt_all" config from conf/dataset_conf/data_path/ccxt_all.yaml
    #"BNB" "AAVE" "ATOM" "ADA" "DOGE" "SOL" "AVAX" "LUNA" "DOT"
)
# ftx data
for SYMBOL in "BTC-PERP" "ETH-PERP"; do
    echo downloading "$SYMBOL"...
    python -- ccxt_downloader.py --symbol="$SYMBOL" --exchange=ftx --start-date "$START_DATE" --end-date "$END_DATE"
done

# # binance data
# for STABLE in "BUSD" "USDT"; do
#     for SYMBOL in ${SYMBOLS[@]}; do
#         PAIR="${SYMBOL}/${STABLE}"
#         echo downloading "$PAIR"...
#         python -- ccxt_downloader.py --symbol="$PAIR" --exchange=binance --start-date "$START_DATE" --end-date "$END_DATE"
#     done
# done

# mkdir high_volume
# ln BTC*.csv ETH*.csv high_volume
