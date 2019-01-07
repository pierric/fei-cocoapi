{-# LANGUAGE ScopedTypeVariables #-}
module MXNet.Coco.Raw where

import Foreign.Storable
import Foreign.Ptr
import Foreign.C.Types
import Foreign.C.String
import Foreign.Marshal.Array
import Foreign.Marshal.Alloc
import Foreign.Marshal.Utils
import Foreign.Storable.Tuple ()
import qualified Data.Vector.Storable as SV
import qualified Data.Vector.Storable.Mutable as SVM
import Control.Exception 

#include "maskApi.h"

data RLE = RLE {
    _rle_h :: Int,
    _rle_w :: Int,
    _rle_m :: Int,
    _rle_cnts :: Ptr CUInt
}

instance Storable RLE where
  sizeOf _ = {#sizeof RLE #}
  alignment _ = 4
  peek p = do
    h <- fromIntegral <$> {#get RLE->h #} p
    w <- fromIntegral <$> {#get RLE->w #} p
    m <- fromIntegral <$> {#get RLE->m #} p
    c <- {#get RLE->cnts #} p
    return $ RLE h w m c
  poke p (RLE h w m c) = do
    {#set RLE.h #} p (fromIntegral h)
    {#set RLE.w #} p (fromIntegral w)
    {#set RLE.m #} p (fromIntegral m)
    {#set RLE.cnts #} p c

allocaRLE :: (Ptr () -> IO a) -> IO a
allocaRLE a = alloca (\ (p :: Ptr RLE) -> a (castPtr p))

peekRLE :: Ptr () -> IO RLE
peekRLE = peek . castPtr

withRLE :: RLE -> (Ptr () -> IO a) -> IO a
withRLE o a = with o (a . castPtr)

withRLEArray :: [RLE] -> (Ptr () -> IO a) -> IO a
withRLEArray o a = withArray o (a . castPtr)

svUnsafeWith :: Storable a => SV.Vector a -> (Ptr a -> IO b) -> IO b
svUnsafeWith = SV.unsafeWith

newtype BB = BB (SV.Vector (CDouble, CDouble, CDouble, CDouble))

{#pointer BB as PtrBB #}

{#fun rleInit as ^ 
    {
        allocaRLE- `RLE' peekRLE*,
        `Int',
        `Int',
        `Int',
        withArray* `[CUInt]'
    } -> `()'
#}

{#fun rleFree as ^
    {
        withRLE* `RLE'
    } -> `()'
#}

{#fun rleEncode as rleEncode_
    {
        `Ptr ()',
        svUnsafeWith* `SV.Vector CUChar',
        `Int',
        `Int',
        `Int'
    } -> `()'
#}
  
rleEncode :: SV.Vector CUChar -> Int -> Int -> Int -> IO [RLE]
rleEncode m h w n = do
    allocaArray n (\ (prle :: Ptr RLE) -> do
        rleEncode_ (castPtr prle) m h w n
        peekArray n prle)

{#fun rleDecode as ^
    {
        withRLE* `RLE',
        svUnsafeWith* `SV.Vector CUChar',
        `Int'
    } -> `()'
#}

{#fun rleMerge as ^ 
    {
        withRLEArray* `[RLE]',
        withRLE* `RLE',
        `Int',
        `Bool'
    } -> `()'
#}

{#fun rleArea as rleArea_
    {
        withRLEArray* `[RLE]',
        `Int',
        id `Ptr CUInt'
    } -> `()'
#}

rleArea :: [RLE] -> Int -> IO [Int]
rleArea r n = do
    allocaArray n (\pa -> do
        rleArea_ r n pa
        map fromIntegral <$> peekArray n pa)
    
{#fun rleIou as rleIou_
    {
        withRLEArray* `[RLE]',
        withRLEArray* `[RLE]',
        `Int',
        `Int',
        svUnsafeWith* `SV.Vector CUChar',
        id `Ptr CDouble'
    } -> `()'
#}

rleIou :: [RLE] -> [RLE] -> [Int] -> IO [Double]
rleIou dt gt iscrowd = do
    let m = length dt
        n = length gt
        c = length iscrowd
    assert (n == c) $ allocaArray (m*n) $ \po -> do
        rleIou_ dt gt m n (SV.fromList $ map fromIntegral iscrowd) po
        map realToFrac <$> peekArray (m * n) po

{#fun rleNms as rleNms_
    {
        withRLEArray* `[RLE]',
        `Int',
        id `Ptr CUInt',
        `CDouble'
    } -> `()'
#}

rleNms :: [RLE] -> Double -> IO [Bool]
rleNms dt thr = do
    let n = length dt
    allocaArray n $ \keep -> do
        rleNms_ dt n keep (realToFrac thr)
        map (>0) <$> peekArray n keep

{#fun bbIou as bbIou_
    {
        `PtrBB',
        `PtrBB',
        `Int',
        `Int',
        svUnsafeWith* `SV.Vector CUChar',
        id `Ptr CDouble'
    } -> `()'
#}

bbIou :: BB -> BB -> [Int] -> IO [Double]
bbIou (BB dt) (BB gt) iscrowd = do
    let m = SV.length dt
        n = SV.length gt
        c = length iscrowd
    assert (n == c) $ allocaArray (m*n) $ \po ->
        svUnsafeWith dt $ \pdt -> svUnsafeWith gt $ \pgt -> do
            bbIou_ (castPtr pdt) (castPtr pgt) m n (SV.fromList $ map fromIntegral iscrowd) po
            map realToFrac <$> peekArray (m * n) po

{#fun bbNms as bbNms_
    {
        `PtrBB',
        `Int',
        id `Ptr CUInt',
        `CDouble'
    } -> `()'
#}

bbNms :: BB -> Double -> IO [Bool]
bbNms (BB dt) thr = do
    let n = SV.length dt
    svUnsafeWith dt $ \pbb -> 
        allocaArray n $ \keep -> do
            bbNms_ (castPtr pbb) n keep (realToFrac thr)
            map (>0) <$> peekArray n keep

{#fun rleToBbox as rleToBbox_
    {
        withRLEArray* `[RLE]',
        `PtrBB',
        `Int'
    } -> `()'
#}

rleToBbox :: [RLE] -> IO BB
rleToBbox r = do
    let n = length r
    mbb <- SVM.new n
    SVM.unsafeWith mbb $ \pbb -> rleToBbox_ r (castPtr pbb) n
    BB <$> SV.freeze mbb

{#fun rleFrBbox as rleFrBbox_
    {
        `Ptr ()',
        `PtrBB',
        `Int',
        `Int',
        `Int'
    } -> `()'
#}

rleFrBbox :: BB -> Int -> Int -> IO [RLE]
rleFrBbox (BB bb) h w = do
    let n = SV.length bb
    allocaArray n $ \(pr :: Ptr RLE) -> svUnsafeWith bb $ \pbb -> do
        rleFrBbox_ (castPtr pr) (castPtr pbb) h w n
        peekArray n pr

{#fun rleFrPoly as rleFrPoly_
    {
        `Ptr ()',
        id `Ptr CDouble',
        `Int',
        `Int',
        `Int'
    } -> `()'
#}

rleFrPoly :: SV.Vector (CDouble, CDouble) -> Int -> Int -> IO RLE
rleFrPoly xy h w = do
    let k = SV.length xy
    allocaRLE $ \prle -> svUnsafeWith xy $ \pxy -> do
        rleFrPoly_ prle (castPtr pxy) k h w
        peekRLE prle 

{#fun rleToString as rleToString_
    {
        withRLE* `RLE'
    } -> `String' peekAndFreeCString*
#}

peekAndFreeCString :: Ptr CChar -> IO String
peekAndFreeCString cstr = do
    hstr <- peekCString cstr
    free cstr
    return hstr

{#fun rleFrString as ^
    {
        allocaRLE- `RLE' peekRLE*,
        `String',
        `Int',
        `Int'
    } -> `()'
#}