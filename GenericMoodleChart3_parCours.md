# Outil de génération des graphiques 

Ce programme utilise le fichier .CSV généré par un sondage Moodle pour consituer des graphiques.


```python
import tlc_split as tlc
import os
import ipywidgets as ipw
from ipywidgets import Button
from IPython.display import HTML
from IPython.display import display
from IPython.display import FileLink
from IPython.display import IFrame

from base64 import b64encode

def on_upload_clicked(change):
    global pdf_file
    uploader.disabled = True
    pdf_file = tlc.generate_charts(change)
    out.clear_output()
    with out:
        # link = f'Click here to download: <a href="{pdf_file}" download="{pdf_file}">{pdf_file}</a>'
        link = f'Click here to download: <a href="{pdf_file}">{pdf_file}</a>'
        display(HTML(link))
    uploader.disabled = False


pdf_file = ''

uploader = ipw.FileUpload(
    accept='*.csv',  # Accepted file extension e.g. '.txt', '.pdf', 'image/*', 'image/*,.pdf'
    multiple=False,  # True to accept multiple files upload else False
    description='Moodle (.csv)',
    layout=ipw.Layout(width="200px")
)

uploader.observe(on_upload_clicked, names='value')

display(ipw.HBox([uploader])) 
out = ipw.Output()
display(out)
```

