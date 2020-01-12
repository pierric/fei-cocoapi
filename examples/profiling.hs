{-# Language OverloadedLabels, FlexibleInstances #-}
import Criterion.Main
import Criterion.Main.Options
import Data.Store
import qualified Data.IntSet as Set
import qualified Data.ByteString as BS
import qualified Data.Vector as V
import qualified Data.Vector.Storable as SV
import qualified Data.Vector.Unboxed as UV
import Data.Array.Repa ((:.)(..), Z (..), fromUnboxed, computeUnboxedP, computeUnboxedS)
import qualified Data.Array.Repa as Repa
import qualified Data.Conduit as C
import qualified Data.Conduit.List as C
import Control.Monad.Trans.Resource
import Control.Monad.Reader
import Control.Lens
import Control.DeepSeq

import qualified Codec.Picture as JUC
import Codec.Picture.Extra
import qualified Codec.Picture.Repa as RPJ

import qualified Graphics.Image as HIP

import MXNet.NN.DataIter.Anchor as Anch
import MXNet.NN.DataIter.Coco as Coco
import MXNet.NN.DataIter.Class
import MXNet.NN.DataIter.Conduit
import MXNet.Base (NDArray(..), mxListAllOpNames, (.&), HMap(..), ArgOf(..))

main_boxes = do
    goodIndices <- BS.readFile "examples/goodIndices.bin" >>= decodeIO :: IO (V.Vector Int)
    gtBoxes     <- BS.readFile "examples/gtBoxes.bin"     >>= decodeIO :: IO (V.Vector (UV.Vector Float))
    anchors     <- BS.readFile "examples/anchors.bin"     >>= decodeIO :: IO (V.Vector (UV.Vector Float))
    goodIndices <- return $ Set.fromList $ V.toList goodIndices   :: IO Set.IntSet
    gtBoxes <- return $ V.map (fromUnboxed (Z:.(5::Int))) gtBoxes
    anchors <- return $ V.map (fromUnboxed (Z:.(4::Int))) anchors

    defaultMain
        [ bench "computeUnboxedP" $ whnfIO $ computeUnboxedP $ overlapMatrix goodIndices gtBoxes anchors
        , bench "computeUnboxedS" $ whnf computeUnboxedS $ overlapMatrix goodIndices gtBoxes anchors
        ]

main_scale_image = do
    let imgFilePath = "/home/jiasen/hdd/dschungel/coco/val2017/000000121242.jpg"
    Right imgjuc <- liftIO (JUC.readImage imgFilePath)
    Right imghip <- liftIO (HIP.readImage imgFilePath :: IO (Either String (HIP.Image HIP.VS HIP.RGB Double)))
    print (HIP.dims imghip)
    defaultMain
        [ bench "scale-img-juicy" $ nfIO $
            let img1 = JUC.convertRGB8 imgjuc
                img2 = scaleBilinear 1024 1024 img1
                img3 = RPJ.imgData (RPJ.convertImage img2 :: RPJ.Img RPJ.RGB)
            in Repa.computeUnboxedP img3

        , bench "scale-img-hip" $ nfIO $
            let img2 = HIP.resize HIP.Bilinear HIP.Edge (1024, 1024) imghip
            in return img2
        ]

main_iter = do
    cocoInst <- coco "/home/jiasen/hdd/dschungel/coco" "val2017"
    let imgIter@(ConduitData _ imgC) = cocoImages cocoInst True
        cocoConf = Coco.Configuration 800 (123.68, 116.779, 103.939) (1, 1, 1)
        anchConf = Anch.Configuration {
            Anch._conf_anchor_scales  = [8, 16, 32],
            Anch._conf_anchor_ratios  = [0.5, 1, 2],
            Anch._conf_allowed_border = 0,
            Anch._conf_fg_num         = 128,
            Anch._conf_batch_num      = 256,
            Anch._conf_fg_overlap     = 0.7,
            Anch._conf_bg_overlap     = 0.3}
    anchors <- runReaderT (Anch.anchors 16 50 50) anchConf
    [data0] <- runResourceT $
        flip runReaderT cocoConf $
        C.runConduit $
        getConduit (takeD 1 imgIter) C..| C.consume
    data0_step1 <- assignAnchors anchConf anchors 50 50 Nothing data0

    let morf = C.transPipe (flip runReaderT cocoConf)
        dataIter1 = morf imgC C..| C.mapM (assignAnchors anchConf anchors 50 50 Nothing)
        dataIter2 = dataIter1 C..| C.chunksOf 1
        dataIter3 = dataIter2 C..| C.mapM toNDArray
        dataIter4 = dataIter1 C..| C.mapM (\x -> toNDArray [x])

    defaultMain
        [ bench "img-iter" $ nfIO $
            runResourceT $
                flip runReaderT cocoConf $
                    C.runConduit $
                        getConduit (takeD 10 imgIter) C..| C.consume
        , bench "assign-anchors" $ nfIO $
             assignAnchors anchConf anchors 50 50 Nothing data0
        , bench "to-ndarray" $ nfIO $ toNDArray [data0_step1]
        , bench "img-iter + assign-anchors" $ nfIO $
            runResourceT $
                C.runConduit $
                    dataIter1 C..| C.take 10
        , bench "img-iter + assign-anchors + chunks" $ nfIO $
            runResourceT $
                C.runConduit $
                    dataIter2 C..| C.take 10
        , bench "img-iter + assign-anchors + chunks + to-ndarray" $ nfIO $
            runResourceT $
                C.runConduit $
                    dataIter3 C..| C.take 10
        ]

main = do
    cocoInst <- coco "/home/jiasen/hdd/dschungel/coco" "val2017"
    let iter = cocoImagesWithAnchors cocoInst
                        (#anchor_scales := [8, 16, 32]
                      .& #anchor_ratios := [0.5, 1, 2]
                      .& #batch_rois    := 256
                      .& #feature_stride:= 16
                      .& #allowed_border:= 0
                      .& #fg_fraction   := 0.6
                      .& #fg_overlap    := 0.7
                      .& #bg_overlap    := 0.3
                      .& #mean          := (123.68, 116.779, 103.939)
                      .& #std           := (1, 1, 1)
                      .& #batch_size    := 1
                      .& #image_size    := 800
                      .& #feature_width := 50
                      .& #feature_height:= 50
                      .& #shuffle       := True
                      .& Nil)


    defaultMain
        [ bench "iter" $ nfIO $ runResourceT $ C.runConduit $ getConduit (takeD 10 iter) C..| C.consume
        ]


