import os


outlist = []
for name in os.listdir(r'C:\Users\a046829713\Desktop\program\RLTrading\SynapseX\Brain\simulation\data'):
    if name.split('-')[0] not in ['BTCUSDT', 'ETHUSDT', 'SOLUSDT', 'XRPUSDT', 'DOGEUSDT', 'SUIUSDT', 'ADAUSDT', 'ENAUSDT', 'LINKUSDT', 'HBARUSDT', 'LTCUSDT', 'XLMUSDT', 'WIFUSDT', 'BNBUSDT', 'ONDOUSDT', 'AAVEUSDT', 'WLDUSDT', 'AVAXUSDT', 'JUPUSDT', 'DOTUSDT', 'TRXUSDT', 'FILUSDT', 'ALGOUSDT', 'ZENUSDT', 'TIAUSDT', 'CRVUSDT', 'AGLDUSDT', 'POPCATUSDT', 'GALAUSDT', 'NEARUSDT']:
        outlist.append(name.split('-')[0])


print(outlist)