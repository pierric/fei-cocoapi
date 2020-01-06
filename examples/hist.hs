{-# LANGUAGE FlexibleContexts, OverloadedStrings, PartialTypeSignatures, GADTs, TypeOperators, OverloadedLabels #-}
import Data.Conduit
import qualified Data.Conduit.Combinators as C
import Control.Lens
import Data.Maybe
import Control.Monad.Reader
import Control.Monad.Trans.Resource
import Control.Monad.IO.Class
import System.IO (hFlush, stdout)
import qualified Data.Vector as V
import Data.Histogram.Fill
import Data.Histogram (Histogram)

import MXNet.NN.DataIter.Coco
import MXNet.NN.DataIter.Conduit
import MXNet.Base (NDArray(..), mxListAllOpNames, (.&), HMap(..), ArgOf(..))
import MXNet.Base.NDArray

main = do
    let conf = Configuration 1024 (123.68, 116.779, 103.939) (1,1,1)
    cocoInst <- coco "/home/jiasen/hdd/dschungel/coco" "train2017"
    flip runReaderT conf $ runResourceT $ do
        -- let dataIter = cocoImagesWithAnchors cocoInst
        --             (\_ -> return (50, 50))
        --             (#batch_size     := 1
        --           .& #long_size      := 1024
        --           .& #mean           := (123.68, 116.779, 103.939)
        --           .& #std            := (1,1,1)
        --           .& #feature_stride := 16
        --           .& #anchor_scales  := [4, 8, 16, 32]
        --           .& #anchor_ratios  := [0.5, 1, 2]
        --           .& #allowed_border := 0
        --           .& #batch_rois     := 256
        --           .& #fg_fraction    := 0.5
        --           .& #fg_overlap     := 0.7
        --           .& #bg_overlap     := 0.3
        --           .& Nil) :: ConduitData ResIO _

        let dataIter = cocoImages cocoInst False
        gtSizes <- forEachD_i dataIter $ \(i, e) -> liftIO $ do
            putStr $ "\r\ESC[K" ++ show i
            hFlush stdout
            return $ V.length $ e ^. _3

        liftIO $ print $ histo 10 $ V.fromList gtSizes

histo :: Int -> V.Vector Int -> Histogram BinInt Int
histo n v = fillBuilder buildr v
  where
    mi = minimum v
    ma = maximum v
    bins = binInt mi n ma
    buildr = mkSimple bins


