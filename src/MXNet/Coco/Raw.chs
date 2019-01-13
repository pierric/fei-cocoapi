{-# LANGUAGE ScopedTypeVariables #-}
module MXNet.Coco.Raw where

import Foreign.Storable
import Foreign.Ptr
import Foreign.ForeignPtr
import Foreign.ForeignPtr.Unsafe (unsafeForeignPtrToPtr)
import Foreign.C.Types
import Foreign.C.String (CString)
import Foreign.Marshal.Array
import Foreign.Marshal.Alloc
import Foreign.Storable.Tuple ()
import qualified Data.Vector.Storable as SV
import qualified Data.Vector.Storable.Mutable as SVM
import qualified Data.ByteString as BS
import Control.Exception 

#include "maskApi.h"

data RLE = RLE {
    _rle_h :: Int,
    _rle_w :: Int,
    _rle_m :: Int,
    _rle_cnts :: ForeignPtr CUInt
}

makeRLE :: (Ptr () -> IO ()) -> IO RLE
makeRLE a = makeRLEs 1 a >>= return . head

makeRLEs :: Int -> (Ptr () -> IO ()) -> IO [RLE]
makeRLEs num a = allocaBytesAligned (num * {#sizeof RLE #}) {#alignof RLE#} (\prle -> do
    a prle
    go num prle [])
  where
    go 0 _ rles = return $ reverse rles
    go n prle rles = do
        rle <- peekRLE prle
        go (n-1) (prle `plusPtr` {#sizeof RLE#}) (rle : rles)

    peekRLE prle = do
        h <- fromIntegral <$> {#get RLE->h #} prle
        w <- fromIntegral <$> {#get RLE->w #} prle
        m <- fromIntegral <$> {#get RLE->m #} prle
        raw_c <- {#get RLE->cnts #} prle
        mgr_c <- newForeignPtr finalizerFree raw_c
        return $ RLE h w m mgr_c

withRLE :: RLE -> (Ptr () -> IO a) -> IO a
withRLE rle = withRLEs [rle]

withRLEs :: [RLE] -> (Ptr () -> IO a) -> IO a
withRLEs rles a = do
    let num = length rles
    allocaBytesAligned (num * {#sizeof RLE#}) {#alignof RLE#} $ \prles -> do
        go prles rles
        ret <- a prles
        mapM_ (touchForeignPtr . _rle_cnts) rles
        return ret
  where 
    go _ [] = return ()
    go prles (rle : nrles) = do
        pokeRLE prles rle
        go (prles `plusPtr` {#sizeof RLE#}) nrles

    -- must touch _rle_cnts after using the prle
    pokeRLE prle (RLE h w m c) = do
        {#set RLE.h #} prle (fromIntegral h)
        {#set RLE.w #} prle (fromIntegral w)
        {#set RLE.m #} prle (fromIntegral m)
        {#set RLE.cnts #} prle (unsafeForeignPtrToPtr c)

svUnsafeWith :: Storable a => SV.Vector a -> (Ptr a -> IO b) -> IO b
svUnsafeWith = SV.unsafeWith

newtype BB = BB (SV.Vector (CDouble, CDouble, CDouble, CDouble))

{#pointer BB as PtrBB #}

{#fun rleInit as rleInit_
    {
        `Ptr ()',
        `Int',
        `Int',
        `Int',
        id `Ptr CUInt'
    } -> `()'
#}

rleInit :: Int -> Int -> [CUInt] -> IO RLE
rleInit h w cnts = do
    makeRLE (\pr -> withArrayLen cnts (\m pc -> rleInit_ pr h w m pc))

-- cause the storage owned by rle to be freed immediately,
-- without not calling the c-api rleFree
rleFree :: RLE -> IO ()
rleFree rle = finalizeForeignPtr (_rle_cnts rle)

{#fun rleEncode as rleEncode_
    {
        `Ptr ()',
        id `Ptr CUChar',
        `Int',
        `Int',
        `Int'
    } -> `()'
#}
  
rleEncode :: BS.ByteString -> Int -> Int -> Int -> IO [RLE]
rleEncode m h w n = do
    makeRLEs n (\ prle ->
        withByteString m (\pm -> do 
            rleEncode_ prle (castPtr pm) h w n))

{#fun rleDecode as rleDecode_
    {
        withRLEs* `[RLE]',
        id `Ptr CUChar',
        `Int'
    } -> `()'
#}

rleDecode :: [RLE] -> Int -> Int -> IO BS.ByteString
rleDecode rles h w = do
    let n = length rles 
        size = n * h * w
    allocaBytes size $ (\ptr -> do
        rleDecode_ rles ptr n
        BS.packCStringLen (castPtr ptr, size))

{#fun rleMerge as ^ 
    {
        withRLEs* `[RLE]',
        withRLE* `RLE',
        `Int',
        `Bool'
    } -> `()'
#}

{#fun rleArea as rleArea_
    {
        withRLEs* `[RLE]',
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
        withRLEs* `[RLE]',
        withRLEs* `[RLE]',
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
        withRLEs* `[RLE]',
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
        withRLEs* `[RLE]',
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
    makeRLEs n $ \prles -> svUnsafeWith bb $ \pbb -> do
        rleFrBbox_ prles (castPtr pbb) h w n

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
    makeRLE $ \prle -> svUnsafeWith xy $ \pxy -> do
        rleFrPoly_ prle (castPtr pxy) k h w

{#fun rleToString as ^
    {
        withRLE* `RLE'
    } -> `BS.ByteString' peekAndFreeCString*
#}

peekAndFreeCString :: Ptr CChar -> IO BS.ByteString
peekAndFreeCString cstr = do
    hstr <- BS.packCString cstr
    free cstr
    return hstr

{#fun rleFrString as rleFrString_
    {
        `Ptr ()',
        withByteString* `BS.ByteString',
        `Int',
        `Int'
    } -> `()'
#}

rleFrString :: BS.ByteString -> Int -> Int -> IO RLE
rleFrString bs h w = do
    makeRLE $ (\pr -> rleFrString_ pr bs h w)

withByteString :: BS.ByteString -> (CString -> IO a) -> IO a
withByteString = BS.useAsCString