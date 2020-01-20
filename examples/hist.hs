{-# LANGUAGE FlexibleContexts, OverloadedStrings, PartialTypeSignatures, GADTs, TypeOperators, OverloadedLabels #-}
import Data.Conduit
import Control.Lens
import Data.Maybe
import Control.Monad.Reader
import Control.Monad.Trans.Resource
import Control.Monad.IO.Class
import System.IO (hFlush, stdout)
import qualified Data.Vector as V
import Data.Histogram.Fill
import Data.Histogram (Histogram)
import qualified Data.Conduit as C
import qualified Data.Conduit.Combinators as C
import qualified Data.Conduit.List as C (catMaybes)

import MXNet.NN.DataIter.Coco
import MXNet.NN.DataIter.Conduit
import MXNet.Base (NDArray(..), mxListAllOpNames, (.&), HMap(..), ArgOf(..))
import MXNet.Base.NDArray

main = do
    let conf = Configuration 1024 (123.68, 116.779, 103.939) (1,1,1)
    cocoInst <- coco "/home/jiasen/hdd/dschungel/coco" "train2017"
    flip runReaderT conf $ runResourceT $ do
        let dataIter = ConduitData (Just 1) $ cocoImages cocoInst False C..| C.mapM (loadImageAndGT cocoInst) C..| C.catMaybes
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


