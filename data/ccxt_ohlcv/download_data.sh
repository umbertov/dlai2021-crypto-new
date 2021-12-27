#!/bin/sh

# # ftx data
# for SYMBOL in "BTC-PERP" "BTC/USD" "ETH-PERP" "ETH/USD" "ADA-PERP" "ADA/USD"; do
#     echo downloading "$SYMBOL"...
#     python -- ccxt_downloader.py --symbol="$SYMBOL" --exchange=ftx
# done

# binance data
for STABLE in "USDT" "BUSD"; do
    for SYMBOL in "BTC" "ETH" "BNB" "DOGE" "ADA" "SOL" "DOT"; do
        PAIR="${SYMBOL}/${STABLE}"
        echo downloading "$PAIR"...
        python -- ccxt_downloader.py --symbol="$PAIR" --exchange=binance
    done
done
