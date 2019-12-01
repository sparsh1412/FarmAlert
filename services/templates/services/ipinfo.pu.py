import ipinfo
access_token = '088099235f51ca'
handler = ipinfo.getHandler(access_token)
details = handler.getDetails()
print(details.city)
details.loc