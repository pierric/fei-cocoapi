{-# LANGUAGE FlexibleContexts, OverloadedStrings, PartialTypeSignatures, GADTs, TypeOperators, OverloadedLabels #-}
import Data.Conduit
import Control.Lens ((^.), _3)
import Data.Maybe
import Control.Monad.Reader
import Control.Monad.Trans.Resource
import Control.Monad.IO.Class
import System.IO (hFlush, stdout)
import qualified Data.Vector as V
import qualified Data.Vector.Storable as SV
import Data.Histogram.Fill
import Data.Histogram (Histogram)
import qualified Data.Conduit as C
import qualified Data.Conduit.Combinators as C
import qualified Data.Conduit.List as C (catMaybes)
import qualified Data.IntMap as IntMap
import Data.IORef
import Data.Array.Repa (index, Z(..), (:.)(..))

import MXNet.NN.DataIter.Coco
import MXNet.NN.DataIter.Conduit
import MXNet.Base (NDArray(..), mxListAllOpNames, (.&), HMap(..), ArgOf(..))
import MXNet.Base.NDArray

main1 = do
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


main = do
    gt_class_count_ref <- newIORef IntMap.empty
    gt_count_ref <- newIORef V.empty

    let conf = Configuration 1024 (123.68, 116.779, 103.939) (1,1,1)
    cocoInst <- coco "/home/jiasen/hdd/dschungel/coco" "train2017"
    flip runReaderT conf $ runResourceT $ do
        runConduit $ cocoImages cocoInst False
            C..| C.mapM (loadImageAndGT cocoInst)
            C..| C.catMaybes
            C..| C.iterM (\e -> liftIO $ do
                let gt = V.map (floor . flip index (Z:.4)) (e ^. _3)
                modifyIORef gt_count_ref (flip V.snoc $ V.length gt)
                let gt_occ = IntMap.fromListWith (+) $ zip (V.toList gt) (repeat 1)
                print $ IntMap.toAscList gt_occ
                modifyIORef gt_class_count_ref (IntMap.unionWith (+) gt_occ))
            C..| C.sinkNull
    putStrLn "----------"

    num_gt <- readIORef gt_count_ref
    print $ histo 10 num_gt
    num_per_class <- IntMap.toAscList <$> readIORef gt_class_count_ref
    print num_per_class
