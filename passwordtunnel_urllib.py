import urllib
print("Password/Endpoint IP for localtunnel is:" ,urllib.request.urlopen('https://ipv4.icanhazip.com').read().decode('utf8').strip("\n"))
