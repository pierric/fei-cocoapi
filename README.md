# Haskell wrapper for CocoAPI

# Image Utility
```
$ stack run imageutils -- --help
Coco Utility

Usage: imageutils [BASEDIR] [SPLIT] COMMAND

Available options:
  BASEDIR                  path to coco
  SPLIT                    data split
  -h,--help                Show this help text

Available commands:
  list                     
  dump     
```

`imageutils` is an helper to list and dump images from the coco dataset. It renders bounding boxes and masks overlays as well.
