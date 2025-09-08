from qlib.data import D
import qlib

def main():
    # Initialize qlib first
    qlib.init(provider_uri="~/.qlib/qlib_data/cn_data", region="cn")

    # Get CSI300 stocks
    instruments = D.instruments(market='csi300')
    stock_list = D.list_instruments(instruments, as_list=True)[:10]  # First 10 stocks

    # Define features
    fields = [
        '$close',           # Close price
        '$volume',          # Volume
        'Ref($close, 1)',   # Previous day close
        'Mean($close, 5)',  # 5-day moving average
        '$high-$low'        # Daily range
    ]

    # Get data for specific time range
    df = D.features(
        instruments=stock_list,
        fields=fields,
        start_time='2021-01-01',
        end_time='2021-06-30',
        freq='day'
    )

    print(df.head())

if __name__ == "__main__":
    main()