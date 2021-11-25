#!/bin/sh

# ftx data
for SYMBOL in "BTC-PERP" "BTC/USD" "ETH-PERP" "ETH/USD" "ADA-PERP" "ADA/USD"; do
    echo downloading "$SYMBOL"...
    ipy -- ccxt_downloader.py --symbol="$SYMBOL" --exchange=ftx
done

# binance data
for SYMBOL in "BTC/BUSD" "ETH/USDT" "ETH/BUSD" "ADA/USDT" "ADA/BUSD"; do
    echo downloading "$SYMBOL"...
    ipy -- ccxt_downloader.py --symbol="$SYMBOL" --exchange=binance
done
