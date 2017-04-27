import time

while True:
    from vahun.LogReader import LogReader
    import pandas as pd
    log=LogReader()
    table=log.get_full_table()
    table.to_csv('/mnt/store/velkey/result_full.tsv',sep='\t')
    print(table)
    time.sleep(60)
