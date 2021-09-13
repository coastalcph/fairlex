import subprocess
from multiprocessing import Pool

def download_zip(jurisdiction):
    partial_command = '''curl --header "Host: case.law" --header "User-Agent: Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.106 Safari/537.36" --header "Accept: text/html,application/xhtml+xml,application/xml;q=0.9,image/avif,image/webp,image/apng,*/*;q=0.8,application/signed-exchange;v=b3;q=0.9" --header "Accept-Language: en-GB,en-US;q=0.9,en;q=0.8" --header "Referer: https://case.law/download/bulk_exports/latest/by_jurisdiction/case_text_restricted/ala/" --header "Cookie: csrftoken=5iFkWeJDzzLSpUFwVD7jkSCIarwDJAoS7eerJnsOvW89oixQiW91JPMBfzup0xsO; sessionid=32x2unqolr0mnfoxy6f04eztb5e5mpyr" --header "Connection: keep-alive" "https://case.law/download/bulk_exports/20200604/by_jurisdiction/case_text_restricted/{}/{}_text_20200604.zip" -L -o "data/case.law/{}_text_20200604.zip"'''
    command = partial_command.format(jurisdiction, jurisdiction, jurisdiction)
    print(f'downloading {jurisdiction}')
    process = subprocess.Popen(command, shell=True, stdout=subprocess.PIPE)
    process.wait()
    unzip_command = 'cd data/case.law; unzip {}_text_20200604.zip'.format(jurisdiction)
    process = subprocess.Popen(unzip_command, shell=True, stdout=subprocess.PIPE)
    process.wait()

def download_case_law():
    jurisdictions = ["ala","alaska","am-samoa","ariz","cal","colo","conn","dakota-territory","dc","del","fla","ga","guam","haw","idaho","ind","iowa","kan","ky","la","mass","md","me","mich","minn","miss","mo","mont","n-mar-i","navajo-nation","nd","neb","nev","nh","nj","ny","ohio","okla","or","pa","pr","ri","sc","sd","tenn","tex","tribal","uk","us","utah","va","vi","vt","w-va","wash","wis","wyo"]
    with Pool(16) as pool:
        res = pool.map_async(download_zip, jurisdictions)
        res.get()
    
        

if __name__=='__main__':
    download_case_law()
    