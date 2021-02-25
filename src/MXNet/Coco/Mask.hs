{-# LANGUAGE OverloadedLists #-}
{-# LANGUAGE TypeOperators   #-}
module MXNet.Coco.Mask where

import qualified Data.Vector.Storable as SV (unsafeCast)
import           RIO
import qualified RIO.ByteString       as BS
import qualified RIO.Vector.Storable  as SV (Vector, fromList, length, map)

import           MXNet.Base           (NDArray, fromVector, ndshape, toVector)
import           MXNet.Coco.Raw

-- NOTE:
-- mask should be of 3 dimension and in CWH order
-- also assuming that CDouble has the same memory representation as Double
type Mask = NDArray Word8  -- DIM3
type Area = NDArray Double -- DIM1
type Iou  = NDArray Double -- DIM2
type BBox = NDArray Double -- DIM2
type Poly = SV.Vector Double

data CompactRLE = CompactRLE Int Int (NonEmpty BS.ByteString)

data CocoError = SizeMismatch
  deriving Show

instance Exception CocoError

encode :: Mask -> IO CompactRLE
encode mask = do
    [n, w, h] <- ndshape mask
    bytes <- SV.map fromIntegral <$> toVector mask
    rles  <- rleEncode bytes h w n
    CompactRLE h w <$> mapM rleToString rles

decode :: CompactRLE -> IO Mask
decode im@(CompactRLE h w bss) = do
    let n = length bss
    rles <- frString im
    raw <- rleDecode rles h w
    fromVector [n, w, h] $ SV.map fromIntegral raw

merge :: CompactRLE -> Bool -> IO CompactRLE
merge im intersect = do
    let CompactRLE h w bss = im
        n = length bss
    if n > 1 then do
        rles <- frString im
        orle <- rleMerge rles intersect
        bs <- rleToString orle
        return $ CompactRLE h w (bs :| [])
    else
        return im

area :: CompactRLE -> IO Area
area im@(CompactRLE _ _ bss) = do
    let num = length bss
    rles <- frString im
    as <- rleArea rles num
    fromVector [num] $ SV.map fromIntegral $ SV.fromList as

iouRLEs :: CompactRLE -> CompactRLE -> [Bool] -> IO Iou
iouRLEs dt gt iscrowd = do
    dt_rles <- frString dt
    gt_rles <- frString gt
    ((m, n), arr) <- rleIou dt_rles gt_rles iscrowd
    fromVector [n, m] $ SV.fromList arr

iouBBs :: BBox -> BBox -> [Bool] -> IO Iou
iouBBs bb1 bb2 iscrowd = do
    bb1' <- BB . SV.unsafeCast <$> toVector bb1
    bb2' <- BB . SV.unsafeCast <$> toVector bb2
    ((m, n), arr) <- bbIou bb1' bb2' iscrowd
    fromVector [n, m] $ SV.fromList arr

toBBox :: CompactRLE -> IO BBox
toBBox im = do
    rles <- frString im
    BB bb <- rleToBbox rles
    fromVector [SV.length bb, 4] (SV.unsafeCast bb)

frBBox :: BBox -> Int -> Int -> IO CompactRLE
frBBox bb h w = do
    bb <- toVector bb
    if SV.length bb /= (h * w * 4)
        then throwM SizeMismatch
        else do
            rles <- rleFrBbox (BB $ SV.unsafeCast bb) h w
            CompactRLE h w <$> mapM rleToString rles

frPoly :: NonEmpty Poly -> Int -> Int -> IO CompactRLE
frPoly polys h w = do
    rles <- mapM (\poly -> rleFrPoly (SV.unsafeCast poly) h w) polys
    CompactRLE h w <$> mapM rleToString rles

frUncompressedRLE :: [Int] -> Int -> Int -> IO CompactRLE
frUncompressedRLE raw h w = do
    orle <- rleInit h w (map fromIntegral raw)
    crle <- rleToString orle
    return $ CompactRLE h w (crle :| [])

frString :: CompactRLE -> IO (NonEmpty RLE)
frString (CompactRLE h w bss) = mapM (\bs -> rleFrString bs h w) bss
