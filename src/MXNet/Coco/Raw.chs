{-# LANGUAGE ScopedTypeVariables #-}
module MXNet.Coco.Raw where

import RIO
import RIO.Partial (toEnum)
import qualified RIO.NonEmpty as RNE
import qualified RIO.NonEmpty.Partial as RNE
import qualified RIO.Vector.Storable as SV
import qualified RIO.Vector.Storable.Unsafe as SV
import qualified Data.Vector.Storable.Mutable as SVM
import qualified RIO.ByteString as BS
import Foreign.Ptr
import Foreign.ForeignPtr
import Foreign.ForeignPtr.Unsafe (unsafeForeignPtrToPtr)
import Foreign.C.Types
import Foreign.C.String (CString)
import Foreign.Marshal.Array
import Foreign.Marshal.Alloc
import Foreign.Storable.Tuple ()

#include "maskApi.h"

data RLE = RLE {
    _rle_h :: Int,
    _rle_w :: Int,
    _rle_m :: Int,
    _rle_cnts :: ForeignPtr CUInt
}

makeRLE :: (Ptr () -> IO ()) -> IO RLE
makeRLE a = makeRLEs 1 a >>= return . RNE.head

makeRLEs :: Int -> (Ptr () -> IO ()) -> IO (NonEmpty RLE)
makeRLEs num a
    | num < 1 = error "number should be positive"
    | otherwise = do
        rles <- allocaBytesAligned (num * {#sizeof RLE #}) {#alignof RLE#} (\prle -> do
                    a prle
                    go num prle [])
        return $ RNE.fromList rles
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
withRLE rle = withRLEs (RNE.fromList [rle])

withRLEs :: NonEmpty RLE -> (Ptr () -> IO a) -> IO a
withRLEs rles = withRLEsLen (length rles) rles

withRLEsLen :: Int -> NonEmpty RLE -> (Ptr () -> IO a) -> IO a
withRLEsLen num rles a = do
    allocaBytesAligned (num * {#sizeof RLE#}) {#alignof RLE#} $ \prles -> do
        go prles $ RNE.toList rles
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

rleEncode :: SV.Vector CUChar -> Int -> Int -> Int -> IO (NonEmpty RLE)
rleEncode m h w n
    | n < 1 = error "number must be positive"
    | otherwise = do
        makeRLEs n (\ prle ->
            svUnsafeWith m (\pm -> do
                rleEncode_ prle (castPtr pm) h w n))

{#fun rleDecode as rleDecode_
    {
        `Ptr ()',
        id `Ptr CUChar',
        `Int'
    } -> `()'
#}

rleDecode :: NonEmpty RLE -> Int -> Int -> IO (SV.Vector CUChar)
rleDecode rles h w = do
    let n = length rles
        size = n * h * w
    mv <- SVM.new size
    SVM.unsafeWith mv $ (\ptr -> do
        withRLEsLen n rles $ \prles -> do
            rleDecode_ prles ptr n)
    SV.unsafeFreeze mv

{#fun rleMerge as rleMerge_
    {
        `Ptr ()',
        `Ptr ()',
        `Int',
        `Bool'
    } -> `()'
#}

rleMerge :: NonEmpty RLE -> Bool -> IO RLE
rleMerge rles intersect = do
    let num = length rles
    withRLEsLen num rles $ \prles ->
        makeRLE $ \porle ->
            rleMerge_ prles porle num intersect

{#fun rleArea as rleArea_
    {
        withRLEs* `NonEmpty RLE',
        `Int',
        id `Ptr CUInt'
    } -> `()'
#}

rleArea :: NonEmpty RLE -> Int -> IO [CUInt]
rleArea r n
    | n < 1 = error "number must be positive"
    | otherwise = do
    allocaArray n (\pa -> do
        rleArea_ r n pa
        peekArray n pa)

{#fun rleIou as rleIou_
    {
        `Ptr ()',
        `Ptr ()',
        `Int',
        `Int',
        svUnsafeWith* `SV.Vector CUChar',
        id `Ptr CDouble'
    } -> `()'
#}

rleIou :: NonEmpty RLE -> NonEmpty RLE -> [Bool] -> IO ((Int,Int), [Double])
rleIou dt gt iscrowd = do
    let m = length dt
        n = length gt
        c = length iscrowd
    assert (n == c) $ allocaArray (m*n) $ \po ->
        withRLEsLen m dt $ \pdt ->
        withRLEsLen n gt $ \pgt -> do
            rleIou_ pdt pgt m n (SV.fromList $ map (toEnum . fromEnum) iscrowd) po
            raw <- peekArray (m * n) po
            return $ ((m,n), map realToFrac raw)

{#fun rleNms as rleNms_
    {
        withRLEs* `NonEmpty RLE',
        `Int',
        id `Ptr CUInt',
        `CDouble'
    } -> `()'
#}

rleNms :: NonEmpty RLE -> Double -> IO [Bool]
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

bbIou :: BB -> BB -> [Bool] -> IO ((Int,Int), [Double])
bbIou (BB dt) (BB gt) iscrowd = do
    let m = SV.length dt
        n = SV.length gt
        c = length iscrowd
    assert (n == c) $ allocaArray (m*n) $ \po ->
        svUnsafeWith dt $ \pdt -> svUnsafeWith gt $ \pgt -> do
            bbIou_ (castPtr pdt) (castPtr pgt) m n (SV.fromList $ map (toEnum . fromEnum) iscrowd) po
            raw <- peekArray (m * n) po
            return $ ((m,n), map realToFrac raw)

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
        withRLEs* `NonEmpty RLE',
        `PtrBB',
        `Int'
    } -> `()'
#}

rleToBbox :: NonEmpty RLE -> IO BB
rleToBbox r = do
    let n = length r
    mbb <- SVM.new n
    SVM.unsafeWith mbb $ \pbb -> rleToBbox_ r (castPtr pbb) n
    BB <$> SV.unsafeFreeze mbb

{#fun rleFrBbox as rleFrBbox_
    {
        `Ptr ()',
        `PtrBB',
        `Int',
        `Int',
        `Int'
    } -> `()'
#}

rleFrBbox :: BB -> Int -> Int -> IO (NonEmpty RLE)
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
