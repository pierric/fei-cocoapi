{-# LANGUAGE TemplateHaskell #-}
module MXNet.NN.DataIter.Coco where

import Data.Word
import System.FilePath
import System.Directory
import GHC.Generics (Generic)
import qualified Data.ByteString as SBS
import qualified Data.Store as Store
import Control.Exception
import Data.Array.Repa hiding ((++))
import qualified Data.Vector as V
import qualified Data.Vector.Unboxed as UV
import qualified Codec.Picture.Repa as RPJ
import Codec.Picture
import Codec.Picture.Types
import Codec.Picture.Extra
import qualified Data.Aeson as Aeson
import Control.Lens ((^.), view, makeLenses)
import Data.Conduit
import qualified Data.Conduit.Combinators as C
import Control.Monad.Reader


import MXNet.NN.DataIter.Conduit
import MXNet.Coco.Types

data Coco = Coco FilePath String Instance
  deriving Generic
instance Store.Store Coco

raiseLeft :: Exception e => (a -> e) -> Either a b -> b
raiseLeft exc = either (throw . exc) id

data FileNotFound = FileNotFound String String
  deriving Show
instance Exception FileNotFound

cached :: Store.Store a => String -> IO a -> IO a
cached name action = do
    createDirectoryIfMissing True "cache"
    hitCache <- doesFileExist path
    if hitCache then
        SBS.readFile path >>= Store.decodeIO
    else do
        obj <- action
        SBS.writeFile path (Store.encode obj)
        return obj
  where
    path = "cache/" ++ name
    
coco :: String -> String -> IO Coco
coco base datasplit = cached (datasplit ++ ".store") $ do
    let annotationFile = base </> "annotations" </> ("instances_" ++ datasplit ++ ".json")
    inst <- raiseLeft (FileNotFound annotationFile) <$> Aeson.eitherDecodeFileStrict' annotationFile
    return $ Coco base datasplit inst

type ImageTensor = Array D DIM3 Word8
type ImageInfo = Array U DIM1 Float
type GTBoxes = Array U DIM2 Float

data Configuration = Configuration {
    _conf_short :: Int,
    _conf_max_size :: Int, 
    _conf_mean :: (Float, Float, Float),
    _conf_std :: (Float, Float, Float)
}
makeLenses ''Configuration

cocoImages :: (MonadReader Configuration m, MonadIO m) => Coco -> ConduitData m (ImageTensor, ImageInfo, GTBoxes)
cocoImages (Coco base datasplit inst) = ConduitData $ C.yieldMany (inst ^. images) .| C.mapM loadImg
  where
    tupleToRepa (a, b, c) = fromListUnboxed (Z :. (3 :: Int)) [a,b,c]
    dropAlpha tensor = 
        let Z :. _ :. w :. h = extent tensor
        in fromFunction (Z :. (3 :: Int) :. w :. h) (tensor !)
    loadImg img = do
        short <- view conf_short
        maxSize <- view conf_max_size
        mean <- view conf_mean
        std <- view conf_std
        let meanRepa = tupleToRepa mean
            stdRepa  = tupleToRepa std
            imgFilePath = base </> datasplit </> img ^. img_file_name
        imgDyn <- raiseLeft (FileNotFound imgFilePath) <$> liftIO (readImage imgFilePath)
        let imgRGB = convertRGB8 imgDyn
            imgH = fromIntegral $ imageHeight imgRGB
            imgW = fromIntegral $ imageWidth imgRGB

            scale = calcScale imgW imgH short maxSize
            imgInfo = fromListUnboxed (Z :. 3) [imgH, imgW, scale]

            imgResized = scaleBilinear (floor $ scale * imgW) (floor $ scale * imgH) imgRGB
            imgRGBRepa = RPJ.imgData (RPJ.convertImage imgResized :: RPJ.Img RPJ.RGB)
            imgTrans = transform imgRGBRepa

            gt_boxes = get_gt_boxes scale img
        return $ (imgTrans, imgInfo, gt_boxes)

    -- find a proper scale factor
    calcScale imgW imgH short maxSize = 
      let imSizeMin = min imgH imgW
          imSizeMax = max imgH imgW
          imScale0 = fromIntegral short / imSizeMin  :: Float
          imScale1 = fromIntegral maxSize / imSizeMax :: Float
      in if round (imScale0 * imSizeMax) > maxSize then imScale1 else imScale0

    -- get all the bbox and gt for the image
    get_gt_boxes scale img = fromUnboxed (Z :. (UV.length gtBoxes `div` 5) :. 5) gtBoxes
      where
        imageId = img ^. img_id
        width   = img ^. img_width
        height  = img ^. img_height
        imgAnns = V.filter (\ann -> ann ^. ann_image_id == imageId) (inst ^. annotations)
        gtBoxes = UV.concat $ V.toList $ V.map makeGTBox imgAnns

        cleanBBox (x, y, w, h) = 
          let x0 = max 0 x
              y0 = max 0 y
              x1 = min (fromIntegral width - 1)  (x0 + max 0 (w-1))
              y1 = min (fromIntegral height - 1) (y0 + max 0 (h-1))
          in (x0, y0, x1, y1)

        makeGTBox ann =
          let (x0,y0,x1,y1) = cleanBBox (ann ^. ann_bbox)
              catId = ann ^. ann_category_id 
          in
          if ann ^. ann_area > 0 && x1 > x0 && y1 > y0 
            then UV.fromList [x0*scale,y0*scale,x1*scale,y1*scale,fromIntegral catId]
            else UV.empty

-- transform HWC -> CHW
transform img = backpermute newShape (\ (Z :. c :. h :. w) -> Z :. h :. w :. c) img
  where 
    (Z :. height :. width :. chn) = extent img
    newShape = Z:. chn :. height :. width

-- transform CHW -> HWC
transformInv img = backpermute newShape (\ (Z :. h :. w :. c) -> Z :. c :. h :. w) img
  where 
    (Z :. chn :. height :. width) = extent img
    newShape = Z :. height :. width :. chn