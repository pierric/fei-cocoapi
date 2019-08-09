{-# LANGUAGE TemplateHaskell #-}
{-# LANGUAGE DataKinds #-}
{-# LANGUAGE ScopedTypeVariables #-}
module MXNet.NN.DataIter.Coco (
    cocoImagesWithAnchors,
    Coco(..),
    coco,
) where

import Data.Maybe (catMaybes, fromMaybe)
import Data.List (unzip6)
import System.FilePath
import System.Directory
import GHC.Generics (Generic)
import qualified Data.ByteString as SBS
import qualified Data.Store as Store
import Control.Exception
import Data.Array.Repa (Array, DIM1, DIM3, D, U, (:.)(..), Z (..), Any(..),
    fromListUnboxed, extent, backpermute, extend, (-^), (+^), (*^), (/^))
import qualified Data.Array.Repa as Repa
import Data.Array.Repa.Repr.Unboxed (Unbox)
import qualified Data.Vector as V
import qualified Data.Vector.Unboxed as UV
import qualified Codec.Picture.Repa as RPJ
import Codec.Picture
import Codec.Picture.Extra
import qualified Data.Aeson as Aeson
import Control.Lens ((^.), (%~) , view, makeLenses, _1, _2)
import Data.Conduit
import qualified Data.Conduit.Combinators as C (yieldMany)
import qualified Data.Conduit.List as C
import Control.Monad.Reader
import qualified Data.IntMap.Strict as M
import Data.Maybe (fromJust)

import MXNet.Base (NDArray(..), Fullfilled, ArgsHMap, ParameterList, Attr(..), (!), (!?), (.&), HMap(..), ArgOf(..), fromVector,zeros)
import MXNet.Base.Operators.NDArray (_Reshape)
import MXNet.NN.DataIter.Conduit
import qualified MXNet.NN.DataIter.Anchor as Anchor
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

type ImageTensor = Array U DIM3 Float
type ImageInfo = Array U DIM1 Float
type GTBoxes = V.Vector (Array U DIM1 Float)

data Configuration = Configuration {
    _conf_short :: Int,
    _conf_max_size :: Int,
    _conf_mean :: (Float, Float, Float),
    _conf_std :: (Float, Float, Float)
}
makeLenses ''Configuration

cocoImages :: (MonadReader Configuration m, MonadIO m) => Coco -> ConduitData m (ImageTensor, ImageInfo, GTBoxes)
cocoImages (Coco base datasplit inst) = ConduitData (Just 1) $ C.yieldMany (inst ^. images) .| C.mapM loadImg
  where
    -- dropAlpha tensor =
    --     let Z :. _ :. w :. h = extent tensor
    --     in fromFunction (Z :. (3 :: Int) :. w :. h) (tensor Repa.!)
    loadImg img = do
        short <- view conf_short
        maxSize <- view conf_max_size

        let imgFilePath = base </> datasplit </> img ^. img_file_name
        imgDyn <- raiseLeft (FileNotFound imgFilePath) <$> liftIO (readImage imgFilePath)

        let imgRGB = convertRGB8 imgDyn
            imgH = fromIntegral $ imageHeight imgRGB
            imgW = fromIntegral $ imageWidth imgRGB

            scale = calcScale imgW imgH short maxSize
            imgH' = floor $ scale * imgH
            imgW' = floor $ scale * imgW
            imgInfo = fromListUnboxed (Z :. 3) [fromIntegral imgH', fromIntegral imgW', scale]

            imgResized = scaleBilinear imgW' imgH' imgRGB
            imgRGBRepa = RPJ.imgData (RPJ.convertImage imgResized :: RPJ.Img RPJ.RGB)

            gt_boxes = get_gt_boxes scale img

        imgTrans <- transform (Repa.map fromIntegral imgRGBRepa)
        imgEval  <- Repa.computeP imgTrans
        return $ (imgEval, imgInfo, gt_boxes)

    -- find a proper scale factor
    calcScale imgW imgH short maxSize =
      let imSizeMin = min imgH imgW
          imSizeMax = max imgH imgW
          imScale0 = fromIntegral short / imSizeMin  :: Float
          imScale1 = fromIntegral maxSize / imSizeMax :: Float
      in if round (imScale0 * imSizeMax) > maxSize then imScale1 else imScale0

    -- map each category from id to its index in the cocoClassNames.
    catTabl = M.fromList $ V.toList $ V.map (\cat -> (cat ^. odc_id, fromJust $ V.elemIndex (cat ^. odc_name) cocoClassNames)) (inst ^. categories)

    -- get all the bbox and gt for the image
    get_gt_boxes scale img = V.fromList $ catMaybes $ map makeGTBox $ V.toList imgAnns
      where
        imageId = img ^. img_id
        width   = img ^. img_width
        height  = img ^. img_height
        imgAnns = V.filter (\ann -> ann ^. ann_image_id == imageId) (inst ^. annotations)

        cleanBBox (x, y, w, h) =
          let x0 = max 0 x
              y0 = max 0 y
              x1 = min (fromIntegral width - 1)  (x0 + max 0 (w-1))
              y1 = min (fromIntegral height - 1) (y0 + max 0 (h-1))
          in (x0, y0, x1, y1)

        makeGTBox ann =
          let (x0, y0, x1, y1) = cleanBBox (ann ^. ann_bbox)
              classId = catTabl M.! (ann ^. ann_category_id)
              
          in
          if ann ^. ann_area > 0 && x1 > x0 && y1 > y0
            then Just $ fromListUnboxed (Z :. 5) [x0*scale, y0*scale, x1*scale, y1*scale, fromIntegral classId]
            else Nothing


-- transform HWC -> CHW
transform :: (Repa.Source r Float, MonadReader Configuration m) =>
    Array r DIM3 Float -> m (Array D DIM3 Float)
transform img = do
    mean <- view conf_mean
    std <- view conf_std
    let broadcast = extend (Any :. height :. width)
        mean' = broadcast $ fromTuple mean
        std'  = broadcast $ fromTuple std
        chnFirst = backpermute newShape (\ (Z :. c :. h :. w) -> Z :. h :. w :. c) img
    return $ (chnFirst -^ mean') /^ std'
  where
    (Z :. height :. width :. chn) = extent img
    newShape = Z:. chn :. height :. width

-- transform CHW -> HWC
transformInv :: (Repa.Source r Float, MonadReader Configuration m) =>
    Array r DIM3 Float -> m (Array D DIM3 Float)
transformInv img = do
    mean <- view conf_mean
    std <- view conf_std
    let broadcast = extend (Any :. height :. width)
        mean' = broadcast $ fromTuple mean
        std'  = broadcast $ fromTuple std
        addMean = img *^ std' +^ mean'
    return $ backpermute newShape (\ (Z :. h :. w :. c) -> Z :. c :. h :. w) addMean
  where
    (Z :. chn :. height :. width) = extent img
    newShape = Z :. height :. width :. chn

fromTuple :: Unbox a => (a, a, a) -> Array U (Z :. Int) a
fromTuple (a, b, c) = fromListUnboxed (Z :. (3 :: Int)) [a,b,c]

cocoClassNames = V.fromList [
    "__background__",  -- always index 0
    "person", "bicycle", "car", "motorcycle", "airplane", "bus", "train",
    "truck", "boat", "traffic light", "fire hydrant", "stop sign",
    "parking meter", "bench", "bird", "cat", "dog", "horse", "sheep",
    "cow", "elephant", "bear", "zebra", "giraffe", "backpack", "umbrella",
    "handbag", "tie", "suitcase", "frisbee", "skis", "snowboard",
    "sports ball", "kite", "baseball bat", "baseball glove", "skateboard",
    "surfboard", "tennis racket", "bottle", "wine glass", "cup", "fork",
    "knife", "spoon", "bowl", "banana", "apple", "sandwich", "orange",
    "broccoli", "carrot", "hot dog", "pizza", "donut", "cake", "chair",
    "couch", "potted plant", "bed", "dining table", "toilet", "tv",
    "laptop", "mouse", "remote", "keyboard", "cell phone", "microwave",
    "oven", "toaster", "sink", "refrigerator", "book", "clock", "vase",
    "scissors", "teddy bear", "hair drier", "toothbrush"]

type instance ParameterList "CocoImagesWithAnchors" =
    '[ '("batch_size",     'AttrReq Int),
       '("short_size",     'AttrReq Int),
       '("long_size",      'AttrReq Int),
       '("mean",           'AttrReq (Float, Float, Float)),
       '("std",            'AttrReq (Float, Float, Float)),
       '("feature_stride", 'AttrOpt Int),
       '("anchor_scales",  'AttrOpt [Int]),
       '("anchor_ratios",  'AttrOpt [Float]),
       '("allowed_border", 'AttrOpt Int),
       '("batch_rois",     'AttrOpt Int),
       '("fg_fraction",    'AttrOpt Float),
       '("fg_overlap",     'AttrOpt Float),
       '("bg_overlap",     'AttrOpt Float)]

cocoImagesWithAnchors :: (Fullfilled "CocoImagesWithAnchors" args, MonadIO m) =>
    Coco -> ((Int, Int) -> IO (Int, Int)) -> ArgsHMap "CocoImagesWithAnchors" args -> 
    ConduitData m ((NDArray Float, NDArray Float, NDArray Float), (NDArray Float, NDArray Float, NDArray Float))
cocoImagesWithAnchors cocoDef extractFeatureShape args = ConduitData (Just batchSize) $ 
    morf imgs .| C.mapM (assignAnchors anchConf) .| C.chunksOf batchSize .| C.mapM toNDArray

  where
    ConduitData _ imgs = cocoImages cocoDef
    cocoConf = Configuration {
        _conf_short    = args ! #short_size,
        _conf_max_size = args ! #long_size,
        _conf_mean     = args ! #mean,
        _conf_std      = args ! #std
    }
    batchSize     = args ! #batch_size
    batchRois = fromMaybe 256 $ args !? #batch_rois
    featureStride = fromMaybe 16 $ args !? #feature_stride
    anchConf = Anchor.Configuration {
        Anchor._conf_anchor_scales  = fromMaybe [8, 16, 32] $ args !? #anchor_scales,
        Anchor._conf_anchor_ratios  = fromMaybe [0.5, 1, 2] $ args !? #anchor_ratios,
        Anchor._conf_allowed_border = fromMaybe 0 $ args !? #allowed_border,
        Anchor._conf_fg_num         = floor $ (fromMaybe 0.5 $ args !? #fg_fraction) * fromIntegral batchRois,
        Anchor._conf_batch_num      = batchRois,
        Anchor._conf_fg_overlap     = fromMaybe 0.7 $ args !? #fg_overlap,
        Anchor._conf_bg_overlap     = fromMaybe 0.3 $ args !? #bg_overlap
    }

    morf = transPipe (flip runReaderT cocoConf)

    assignAnchors :: MonadIO m => Anchor.Configuration -> (ImageTensor, ImageInfo, GTBoxes) ->
        m (ImageTensor, ImageInfo, GTBoxes, Repa.Array U DIM1 Float, Repa.Array U DIM3 Float, Repa.Array U DIM3 Float) 
    assignAnchors conf (img, info, gt) = do
        let imHeight = floor $ info Anchor.#! 0
            imWidth  = floor $ info Anchor.#! 1
        (featureWidth, featureHeight) <- liftIO $ extractFeatureShape (imWidth, imHeight)
        anchors <- runReaderT (Anchor.anchors featureStride featureWidth featureHeight) anchConf

        (lbls, targets, weights) <- runReaderT (Anchor.assign gt imWidth imHeight anchors) conf

        -- reshape and transpose labls   from (feat_h * feat_w * #anch,  ) to (#anch,     feat_h, feat_w)
        -- reshape and transpose targets from (feat_h * feat_w * #anch, 4) to (#anch * 4, feat_h, feat_w)
        -- reshape and transpose weights from (feat_h * feat_w * #anch, 4) to (#anch * 4, feat_h, feat_w)
        let numAnch = length (anchConf ^. Anchor.conf_anchor_scales) * length (anchConf ^. Anchor.conf_anchor_ratios)
        lbls    <- Repa.computeP $ 
                    Repa.reshape (Z :. numAnch * featureHeight * featureWidth) $ 
                    Repa.transpose $ 
                    Repa.reshape (Z :. featureHeight * featureWidth :. numAnch) lbls
        targets <- Repa.computeP $ 
                    Repa.reshape (Z :. numAnch * 4 :. featureHeight :. featureWidth) $
                    Repa.transpose $
                    Repa.reshape (Z :. featureHeight * featureWidth :. numAnch * 4) targets
        weights <- Repa.computeP $ 
                    Repa.reshape (Z :. numAnch * 4 :. featureHeight :. featureWidth) $
                    Repa.transpose $
                    Repa.reshape (Z :. featureHeight * featureWidth :. numAnch * 4) weights

        -- let numAnch = length (anchConf ^. Anchor.conf_anchor_scales) * length (anchConf ^. Anchor.conf_anchor_ratios)
        --     cvt1 (Z :. i :. h :. w) = Z :. ((h * featureWidth + w) * numAnch + i)
        --     cvt2 (Z :. i :. h :. w) = let (m, n) = divMod i 4
        --                               in Z :. ((h * featureWidth + w) * numAnch + m) :. n
        -- lbls    <- Repa.computeP $ Repa.reshape (Z :. numAnch * featureHeight :. featureWidth) $ 
        --                 Repa.backpermute (Z :. numAnch :. featureHeight :. featureWidth) cvt1 lbls
        -- targets <- Repa.computeP $ Repa.backpermute (Z :. numAnch * 4 :. featureHeight :. featureWidth) cvt2 targets
        -- weights <- Repa.computeP $ Repa.backpermute (Z :. numAnch *4  :. featureHeight :. featureWidth) cvt2 weights 

        return (img, info, gt, lbls, targets, weights)

    -- toNDArray :: [((ImageTensor, ImageInfo, GTBoxes, Anchor.Labels, Anchor.Targets, Anchor.Weights))] ->
    --     IO ((NDArray Float, NDArray Float, NDArray Float), (NDArray Float, NDArray Float, NDArray Float))
    toNDArray dat = liftIO $ do
        imagesC  <- convertToMX images
        infosC   <- convertToMX infos
        gtboxesC <- convertToMX [if V.null gt then emtpyGT else convertToRepa (V.toList gt) | gt <- gtboxes]
        labelsC  <- convertToMX labels
        targetsC <- convertToMX targets
        weightsC <- convertToMX weights
        return ((imagesC, infosC, gtboxesC), (labelsC, targetsC, weightsC))
      where
        (images, infos, gtboxes, labels, targets, weights) = unzip6 dat
        emtpyGT = Repa.fromUnboxed (Z:.0:.5) UV.empty

        convert :: Repa.Shape sh => [Array U sh Float] -> ([Int], UV.Vector Float)
        convert xs = assert (not (null xs)) $ (ext, vec)
          where
            vec = UV.concat $ map Repa.toUnboxed xs
            sh0 = Repa.extent (head xs)
            ext = length xs : reverse (Repa.listOfShape sh0)
            
        convertToMX :: Repa.Shape sh => [Array U sh Float] -> IO (NDArray Float)
        convertToMX   = uncurry fromVector . (_2 %~ UV.convert) . convert

        -- shape, at the type level, are sequence of Int, although we wnat to append
        -- a dimension at the head, we add Int at the tail, they are the same.
        convertToRepa :: Repa.Shape sh => [Array U sh Float] -> Array U (sh :. Int) Float
        convertToRepa = uncurry Repa.fromUnboxed . (_1 %~ Repa.shapeOfList . reverse) . convert
