module Main where

import qualified Data.ByteString.Lazy as BS
import qualified Data.ByteString as SBS
import Control.Lens ((^.), (^?), ix)
import qualified Data.Vector as V
import qualified Data.Vector.Storable as SV
import qualified Data.Vector.Unboxed as UV
import qualified Data.Aeson as Aeson
import qualified Data.Array.Repa as RP
import Data.Array.Repa ((:.)(..), Z(..))
import qualified Data.Array.Repa.Repr.ForeignPtr as RF
import Codec.Picture as JP
import Codec.Picture.Repa
import qualified Data.Store as Store
import Control.Exception.Base

import MXNet.Coco.Mask
import MXNet.Coco.Types
import MXNet.Coco.Index

data Y8

class ToDynamicImage a where
    toDynamicImage :: Img a -> DynamicImage

instance ToDynamicImage Y8 where
    toDynamicImage (Img arr0) = ImageY8 $ JP.Image w h (SV.unsafeFromForeignPtr0 (RF.toForeignPtr arr) (h*w*z) )
      where 
        (Z :. h :. w :. z) = RP.extent arr
        arr = RP.computeS arr0    
 
readFromCache path = do
    bs <- SBS.readFile path
    Store.decodeIO bs

readFromJson path = do
    bs <- BS.readFile path
    case Aeson.decode' bs of
        Nothing -> error $ "cannot parse annotation file: " ++ path
        Just inst -> return inst

store path obj = do 
    SBS.writeFile path (Store.encode obj)
    return obj

readAnnotations path =
    readFromCache cache_file `catch` (\ e -> do
        let _ = e :: IOException
        readFromJson path >>= 
            store cache_file)
  where
    cache_file = "./instance.store"

annotatino_file = "/home/jiasen/dschungel/coco/annotations/instances_train2017.json"

main = do
    inst <- readAnnotations annotatino_file
    mapM_ (\cat -> putStrLn $ cat ^. odc_name) $ allCats inst 
    let anno = V.head $ allAnns inst
        imgId = anno ^. ann_image_id
        img   = V.head $ V.filter (\img -> img ^. img_id == imgId) (inst ^. images)
        height = img ^. img_height
        width  = img ^. img_width
    
    store "./imgs.store" $ inst ^. images

    store "./anns.store" $ inst ^. annotations
    print (width, height)

    -- putStrLn $ show imgId

    -- crle <- case anno ^. ann_segmentation of 
    --     SegRLE cnts _ -> frUncompressedRLE cnts height width
    --     SegPolygon polys -> frPoly (map SV.fromList polys) height width

    -- mask <- decode crle

    -- let Z :. c :. w :. h = RP.extent mask
    --     maskHW = RP.backpermute (Z :. h :. w :. c) (\ (Z :. c :. w :. h) -> Z :. h :. w :. c) mask
    --     maskImg = toDynamicImage $ (Img $ RP.map (*255) maskHW :: Img Y8)

    -- savePngImage "a.png" maskImg

    -- putStrLn $ img ^. img_file_name
    -- putStrLn $ img ^. img_flickr_url
    -- putStrLn $ img ^. img_coco_url
