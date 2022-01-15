#!/bin/sh

START_DATE="2017-01-01"
END_DATE="2022-01-10"

SYMBOLS=(
    "BTC" "ETH"
    "BNB" "AAVE" "ATOM" "ADA" "DOGE" "SOL" "AVAX" "LUNA" "DOT"
)
# # ftx data
# for SYMBOL in "BTC-PERP" "BTC/USD" "ETH-PERP" "ETH/USD" "ADA-PERP" "ADA/USD"; do
#     echo downloading "$SYMBOL"...
#     python -- ccxt_downloader.py --symbol="$SYMBOL" --exchange=ftx
# done

# binance data
for STABLE in "BUSD" "USDT"; do
    for SYMBOL in ${SYMBOLS[@]}; do
        PAIR="${SYMBOL}/${STABLE}"
        echo downloading "$PAIR"...
        python -- ccxt_downloader.py --symbol="$PAIR" --exchange=binance --start-date "$START_DATE" --end-date "$END_DATE"
    done
done

mkdir high_volume
ln BTC*.csv ETH*.csv high_volume
