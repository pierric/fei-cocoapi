{-# LANGUAGE TemplateHaskell #-}
{-# LANGUAGE DataKinds #-}
{-# LANGUAGE ScopedTypeVariables #-}
module MXNet.NN.DataIter.Coco (
    cocoImages,
    cocoImagesWithAnchors,
    loadImageAndGT,
    Coco(..),
    coco,
    Configuration(..),
    ImageTensor, ImageInfo, assignAnchors, toNDArray,
) where

import Data.Maybe (catMaybes, fromMaybe)
import Data.List (unzip6)
import System.FilePath
import System.Directory
import GHC.Generics (Generic)
import GHC.Float (double2Float)
import qualified Data.ByteString as SBS
import qualified Data.Store as Store
import Control.Exception
import Data.Array.Repa (Array, DIM1, DIM3, D, U, (:.)(..), Z (..), Any(..),
    fromListUnboxed, extent, backpermute, extend, (-^), (+^), (*^), (/^))
import qualified Data.Array.Repa as Repa
import Data.Array.Repa.Repr.Unboxed (Unbox)
import qualified Data.Vector as V
import qualified Data.Vector.Storable as SV
import qualified Graphics.Image as HIP
import qualified Graphics.Image.Interface as HIP
import qualified Data.Aeson as Aeson
import Control.Lens ((^.), view, makeLenses)
import Data.Conduit
import qualified Data.Conduit.Combinators as C (yieldMany)
import qualified Data.Conduit.List as C
import Control.Monad.Reader
import qualified Data.IntMap.Strict as M
import Data.Maybe (fromJust)
import qualified Data.Random as RND (shuffleN, runRVar, StdRandom(..))
import Data.Conduit.ConcurrentMap (concurrentMapM_numCaps)
import Control.Monad.IO.Unlift
import Control.Monad.Trans.Resource
import Control.DeepSeq

import MXNet.Base (NDArray(..), Fullfilled, ArgsHMap, ParameterList, Attr(..), (!), (!?), (.&), HMap(..), ArgOf(..), fromVector)
import MXNet.Base.Operators.NDArray (stack)
import MXNet.NN.DataIter.Conduit
import qualified MXNet.NN.DataIter.Anchor as Anchor
import MXNet.Coco.Types

instance (Repa.Shape sh, Unbox e) => NFData (Array U sh e) where
    rnf arr = Repa.deepSeqArray arr ()

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
    _conf_width :: Int,
    _conf_mean :: (Float, Float, Float),
    _conf_std :: (Float, Float, Float)
}
makeLenses ''Configuration

cocoImages :: MonadIO m => Coco -> Bool -> ConduitT () Image m ()
cocoImages (Coco _ _ inst) shuffle = do
    all_images <- return $ inst ^. images
    all_images <- if shuffle then
                    liftIO $ RND.runRVar (RND.shuffleN (length all_images) (V.toList all_images)) RND.StdRandom
                  else
                    return $ V.toList all_images
    C.yieldMany all_images

loadImageAndGT :: (MonadReader Configuration m, MonadIO m) => Coco -> Image -> m (Maybe (ImageTensor, ImageInfo, GTBoxes))
loadImageAndGT (Coco base datasplit inst) img = do
    width <- view conf_width

    let imgFilePath = base </> datasplit </> img ^. img_file_name
    imgRGB <- raiseLeft (FileNotFound imgFilePath) <$> liftIO (HIP.readImageExact HIP.JPG imgFilePath)

    let (imgH, imgW) = HIP.dims (imgRGB :: HIP.Image HIP.VS HIP.RGB Double)
        imgH_  = fromIntegral imgH
        imgW_  = fromIntegral imgW
        width_ = fromIntegral width
        (scale, imgW', imgH') = if imgW >= imgH
            then (width_ / imgW_, width, floor (imgH_ * width_ / imgW_))
            else (width_ / imgH_, floor (imgW_ * width_ / imgH_), width)
        imgInfo = fromListUnboxed (Z :. 3) [fromIntegral imgH', fromIntegral imgW', scale]

        imgResized = HIP.resize HIP.Bilinear HIP.Edge (imgH', imgW') imgRGB
        imgPadded  = HIP.canvasSize (HIP.Fill $ HIP.PixelRGB 0 0 0) (width, width) imgResized
        imgRepa    = Repa.fromUnboxed (Z:.width:.width:.3) $ SV.convert $ SV.unsafeCast $ HIP.toVector imgPadded
        gt_boxes   = get_gt_boxes scale img

    if V.null gt_boxes
        then return Nothing
        else do
            imgEval <- transform $ Repa.map double2Float imgRepa
            -- deepSeq the array so that the workload are well parallelized.
            return $!! Just (Repa.computeUnboxedS imgEval, imgInfo, gt_boxes)
  where
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
transform :: (MonadReader Configuration m, Repa.Source r Float) =>
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
       -- images are scaled and padded to this size
       '("image_size",     'AttrReq Int),
       '("mean",           'AttrReq (Float, Float, Float)),
       '("std",            'AttrReq (Float, Float, Float)),
       -- anchors are generated on feature image with a stride
       '("feature_width",  'AttrReq Int),
       '("feature_height", 'AttrReq Int),
       '("feature_stride", 'AttrOpt Int),

       '("anchor_scales",  'AttrOpt [Int]),
       '("anchor_ratios",  'AttrOpt [Float]),
       '("allowed_border", 'AttrOpt Int),
       '("batch_rois",     'AttrOpt Int),
       '("fg_fraction",    'AttrOpt Float),
       '("fg_overlap",     'AttrOpt Float),
       '("bg_overlap",     'AttrOpt Float),
       '("shuffle",        'AttrOpt Bool),
       '("fixed_num_gt",   'AttrOpt (Maybe Int))]


cocoImagesWithAnchors :: Fullfilled "CocoImagesWithAnchors" args =>
    Coco -> ArgsHMap "CocoImagesWithAnchors" args ->
    ConduitData (ResourceT IO) ((NDArray Float, NDArray Float, NDArray Float), (NDArray Float, NDArray Float, NDArray Float))
cocoImagesWithAnchors cocoInst args = ConduitData (Just 1) $ do
    anchors <- runReaderT (Anchor.anchors featureStride featW featH) anchConf
    images
        .| concurrentMapM_numCaps 16 (flip runReaderT imgsConf . loadImageAndGT cocoInst)
        .| C.catMaybes
        .| C.mapM (assignAnchors anchConf anchors featW featH maxGT)
        .| C.chunksOf batchSize
        .| C.mapM toNDArray
  where
    shuffle = fromMaybe True $ args !? #shuffle
    images  = cocoImages cocoInst shuffle
    batchSize = args ! #batch_size
    batchRois     = fromMaybe 256 $ args !? #batch_rois
    featW = args ! #feature_width
    featH = args ! #feature_height
    featureStride = fromMaybe 16 $ args !? #feature_stride
    maxGT = fromMaybe Nothing $ args !? #fixed_num_gt
    anchConf = Anchor.Configuration {
        Anchor._conf_anchor_scales  = fromMaybe [8, 16, 32] $ args !? #anchor_scales,
        Anchor._conf_anchor_ratios  = fromMaybe [0.5, 1, 2] $ args !? #anchor_ratios,
        Anchor._conf_allowed_border = fromMaybe 0 $ args !? #allowed_border,
        Anchor._conf_fg_num         = floor $ (fromMaybe 0.5 $ args !? #fg_fraction) * fromIntegral batchRois,
        Anchor._conf_batch_num      = batchRois,
        Anchor._conf_fg_overlap     = fromMaybe 0.7 $ args !? #fg_overlap,
        Anchor._conf_bg_overlap     = fromMaybe 0.3 $ args !? #bg_overlap
    }
    imgsConf = Configuration {
        _conf_width    = args ! #image_size,
        _conf_mean     = args ! #mean,
        _conf_std      = args ! #std
    }

assignAnchors :: MonadIO m =>
    Anchor.Configuration ->
    V.Vector (Anchor.Anchor U) ->
    Int -> Int -> Maybe Int ->
    (ImageTensor, ImageInfo, GTBoxes) ->
    m (ImageTensor, ImageInfo, GTBoxes, Repa.Array U DIM1 Float, Repa.Array U DIM3 Float, Repa.Array U DIM3 Float)
assignAnchors conf anchors featureWidth featureHeight maxGT (img, info, gt) = do
    let imHeight = floor $ info Anchor.#! 0
        imWidth  = floor $ info Anchor.#! 1
    (lbls, targets, weights) <- runReaderT (Anchor.assign gt imWidth imHeight anchors) conf

    -- reshape and transpose labls   from (feat_h * feat_w * #anch,  ) to (#anch,     feat_h, feat_w)
    -- reshape and transpose targets from (feat_h * feat_w * #anch, 4) to (#anch * 4, feat_h, feat_w)
    -- reshape and transpose weights from (feat_h * feat_w * #anch, 4) to (#anch * 4, feat_h, feat_w)
    let numAnch = length (conf ^. Anchor.conf_anchor_scales) * length (conf ^. Anchor.conf_anchor_ratios)
    lbls    <- return $ Repa.computeS $
                Repa.reshape (Z :. numAnch * featureHeight * featureWidth) $
                Repa.transpose $
                Repa.reshape (Z :. featureHeight * featureWidth :. numAnch) lbls
    targets <- return $ Repa.computeS $
                Repa.reshape (Z :. numAnch * 4 :. featureHeight :. featureWidth) $
                Repa.transpose $
                Repa.reshape (Z :. featureHeight * featureWidth :. numAnch * 4) targets
    weights <- return $ Repa.computeS $
                Repa.reshape (Z :. numAnch * 4 :. featureHeight :. featureWidth) $
                Repa.transpose $
                Repa.reshape (Z :. featureHeight * featureWidth :. numAnch * 4) weights

    -- optionally extend gt to a fixed number (padding with 0s)
    gtRet <- case maxGT of
      Nothing -> return gt
      Just maxGT -> do
        let numGT = V.length gt
            nullGT = fromListUnboxed (Z:.5) [0, 0, 0, 0, 0]
        if numGT <= maxGT then
            return $ gt V.++ V.replicate (maxGT - numGT) nullGT
        else
            return $ V.take maxGT gt

    return $!! (img, info, gtRet, lbls, targets, weights)

toNDArray :: MonadIO m =>
    [((ImageTensor, ImageInfo, GTBoxes, Array U DIM1 Float, Array U DIM3 Float, Array U DIM3 Float))] ->
    m ((NDArray Float, NDArray Float, NDArray Float), (NDArray Float, NDArray Float, NDArray Float))
toNDArray dat = liftIO $ do
    imagesC  <- convertToMX images
    infosC   <- convertToMX infos
    gtboxesC <- mapM (convertToMX . V.toList) gtboxes >>= stackList
    labelsC  <- convertToMX labels
    targetsC <- convertToMX targets
    weightsC <- convertToMX weights
    return $!! ((imagesC, infosC, gtboxesC), (labelsC, targetsC, weightsC))
  where
    (images, infos, gtboxes, labels, targets, weights) = unzip6 dat

    stackList arrs = do
        let hdls = map unNDArray arrs
        NDArray . head <$> stack (#data := hdls .& #num_args := length hdls .& Nil)

    repaToNDArray :: Repa.Shape sh => Array U sh Float -> IO (NDArray Float)
    repaToNDArray arr = do
        let sh = reverse $ Repa.listOfShape $ Repa.extent arr
        fromVector sh $ SV.convert $ Repa.toUnboxed arr

    convertToMX arr = mapM repaToNDArray arr >>= stackList
