## Notes

### Split pdf into separate files
*Not really needed*
```
mkdir separated
pdfseparate poules-ek-2020/poules-ek-2020_1.pdf ./separates/poule-%d.pdf
```

### Convert pdf to png
```
mkdir poule-pngs
pdftoppm poules-ek-2020/poules-ek-2020_1.pdf ./poule-pngs/poule -png -r 300
```
