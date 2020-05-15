{-# LANGUAGE TypeOperators #-}
module MXNet.Coco.Mask where

import           Data.Array.Repa              ((:.) (..), Array, DIM1, DIM2,
                                               DIM3, Z (..), extent)
import           Data.Array.Repa.Repr.Unboxed
import qualified Data.Vector.Storable         as SV (unsafeCast)
import           RIO
import qualified RIO.ByteString               as BS
import qualified RIO.Vector.Storable          as SV (Vector, map)
import qualified RIO.Vector.Unboxed           as UV (convert, length)

import           MXNet.Coco.Raw

-- NOTE:
-- mask should be of 3 dimension and in CWH order
-- also assuming that CDouble has the same memory representation as Double
type Mask = Array U DIM3 Word8
type Area = Array U DIM1 Word32
type Iou  = Array U DIM2 Double
type BBox = Array U DIM2 Double
type Poly = SV.Vector Double

data CompactRLE = CompactRLE Int Int (NonEmpty BS.ByteString)

encode :: Mask -> IO CompactRLE
encode mask = do
    let Z :. n :. w :. h = extent mask
    -- assuming Word8 are identical with CUChar
    rles <- rleEncode (SV.map fromIntegral $ UV.convert $ toUnboxed mask) h w n
    CompactRLE h w <$> mapM rleToString rles

decode :: CompactRLE -> IO Mask
decode im@(CompactRLE h w bss) = do
    let n = length bss
    rles <- frString im
    raw <- rleDecode rles h w
    return $
        fromUnboxed (Z :. n :. w :. h) $
        UV.convert $
        SV.map fromIntegral raw

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
    -- assuming the area can be represented as Word32
    return $
        fromListUnboxed (Z :. num) $
        map fromIntegral $ as

iouRLEs :: CompactRLE -> CompactRLE -> [Bool] -> IO Iou
iouRLEs dt gt iscrowd = do
    dt_rles <- frString dt
    gt_rles <- frString gt
    ((m, n), arr) <- rleIou dt_rles gt_rles iscrowd
    return $ fromListUnboxed (Z :. n :. m) arr

iouBBs :: BBox -> BBox -> [Bool] -> IO Iou
iouBBs bb1 bb2 iscrowd = do
    let bb1' = BB $ SV.unsafeCast $ UV.convert $ toUnboxed bb1
        bb2' = BB $ SV.unsafeCast $ UV.convert $ toUnboxed bb2
    ((m, n), arr) <- bbIou bb1' bb2' iscrowd
    return $ fromListUnboxed (Z :. n :. m) arr

toBBox :: CompactRLE -> IO BBox
toBBox im = do
    rles <- frString im
    BB bb <- rleToBbox rles
    let bb' = UV.convert $ SV.unsafeCast  bb
    return $ fromUnboxed (Z :. UV.length bb' :. 4) bb'

frBBox :: BBox -> Int -> Int -> IO CompactRLE
frBBox bb h w = do
    rles <- rleFrBbox (BB $ SV.unsafeCast $ UV.convert $ toUnboxed bb) h w
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
