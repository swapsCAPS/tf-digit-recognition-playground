## TODO
- [x] Implement https://learnopencv.com/feature-based-image-alignment-using-opencv-c-python/  
      This seems like a far better approach than using hard coded features, like the top and bottom lines.
- [x] Fix line straightening approach by adding some margin to the analyzed columns in `findWhitePixel()`
- [ ] Use row range based pre filtering to find input fields

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
