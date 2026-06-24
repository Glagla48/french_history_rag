from bs4 import BeautifulSoup
import requests
import os

base_url = "https://fr.wikipedia.org/"
portailFrance = base_url + "wiki/Portail:France"
header = {"User-Agent": "French_history_bot/0.1, (https://github.com/Glagla48)",
        "Allow": "/",
        "Accept-Encoding": "gzip"}
req = requests.get(portailFrance, headers=header)

soup = BeautifulSoup(req.content, "html.parser")

histoire_div = 0
for d in soup.find_all("div", class_="boite-coloree-alternative portail-france-boite-bleue"):
    if d.find("h2", id="Histoire"):
        histoire_div = d
        break

all_a = [a["href"] for a in histoire_div.find_all("a", href=True)]

docs = os.listdir("data")
lang = "french"
#lang = "english"
count = 0

for a in all_a:
    a_name = a.split("/")[0]
    if not a_name in docs:
        combined_url = base_url + a
        try:
            tmp_req = requests.get(combined_url, headers=header)
            tmp_soup = BeautifulSoup(tmp_req.content, "html.parser")
            tmp_title_tag = tmp_soup.find("h1", id="firstHeading")
            tmp_content_tag = tmp_soup.find("div", class_="mw-body-content")

            if not tmp_title_tag or not tmp_content_tag:
               #Le titre (h1#firstHeading) n'a pas été trouvé sur la page
               # Le contenu (div.mw-body-content) n'a pas été trouvé sur la page
                continue
            tmp_title = tmp_title_tag.text
            
            tmp_content = tmp_content_tag.text

            with open("data/raw/" + lang + "/" + tmp_title + ".txt", "w") as f:
                f.writelines([tmp_title, tmp_content])
        except requests.exceptions.RequestException as e:
            print(f"Erreur lors de la requête HTTP : {e}")
        except ValueError as e:
            print(f"Erreur de parsing : {e}")
        except Exception as e:
            print(f"Une erreur inattendue est survenue : {e}")
    else:
        count += 1

print("Pages esquivée: ", count)
print("Pages scrappées: ", len(all_a) - count)