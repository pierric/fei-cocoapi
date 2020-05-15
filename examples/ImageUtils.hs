{-# LANGUAGE ViewPatterns #-}
module Main where

import qualified Codec.Picture                 as G
import           Control.Lens                  (ix, (^?))
import qualified Data.Aeson                    as Aeson
import           Data.Array.Repa               ((:.) (..), Z (..))
import qualified Data.Array.Repa               as RP
import           Data.Colour.Palette.ColorSet  (infiniteWebColors)
import           Data.Colour.SRGB              as COLOR
import qualified Data.Store                    as Store
import           Formatting
import qualified Graphics.Image                as G (Pixel (PixelY), VS (..),
                                                     displayImage, exchange,
                                                     fromJPImageRGBA8,
                                                     toJPImageY8)
import           Graphics.Image.Interface.Repa (fromRepaArrayP)
import qualified Graphics.Rasterific           as G
import qualified Graphics.Rasterific.Texture   as G
import           Graphics.Text.TrueType        as G
import           Options.Applicative
import           RIO
import qualified RIO.ByteString                as SBS
import qualified RIO.ByteString.Lazy           as LBS
import           RIO.Directory
import           RIO.FilePath
import qualified RIO.HashMap                   as M
import qualified RIO.NonEmpty                  as RNE
import qualified RIO.Vector.Boxed              as V
import qualified RIO.Vector.Storable           as SV
import qualified RIO.Vector.Unboxed            as UV

import           MXNet.Coco.Index
import           MXNet.Coco.Mask
import           MXNet.Coco.Types              hiding (info)


data ArgSpec = ListImages
    | DumpImage
    { _arg_image_id :: Int
    }
    deriving Show

argspec :: Parser (Maybe String, Maybe String, ArgSpec)
argspec = liftA3 (,,)
            (option auto (metavar "BASEDIR" <> value Nothing <> help "path to coco"))
            (option auto (metavar "SPLIT" <> value Nothing <> help "data split"))
            (subparser (listImg <> dumpImg))
    where
        listImg = command "list" $ info (pure ListImages) mempty
        dumpImg = command "dump" $ info
            (DumpImage <$> argument auto (metavar "IMAGE_ID" <> help "image id") <**> helper) mempty


main = do
    (base_user, split_user, argspec) <- execParser $ info (argspec <**> helper) (fullDesc <> header "Coco Utility")
    home <- getHomeDirectory
    let base = fromMaybe (home </> ".mxnet/datasets/coco") base_user
        split = fromMaybe "train2017" split_user
        anno_filename = formatToString ("instances_" % string % ".json") split
    runSimpleApp $ do
        cocoinst <- liftIO $ readAnnotations $ base </> "annotations" </> anno_filename
        case argspec of
            ListImages    -> listImages cocoinst
            DumpImage{..} -> dumpImage cocoinst base split _arg_image_id

listImages cocoinst = do
    logInfo $ display ("ImageID        Height  Width   Filename" :: Text)
    forM_ (cocoinst ^. images) $ \image -> do
        logInfo . display $ sformat
            (right 15 ' ' % right 8 ' ' % right 8 ' ' % fitRight 80)
            (image ^. img_id) (image ^. img_height) (image ^. img_width) (image ^. img_file_name)

dumpImage cocoinst base split imgid = do
    let annos = V.filter (\ann -> ann ^. ann_image_id == imgid) $ cocoinst ^. annotations
        image = V.filter (\img -> img ^. img_id == imgid) $ cocoinst ^. images
        cat_table = M.fromList $ V.toList $
            V.map (\cat -> (cat ^. odc_id, cat ^. odc_name)) $ cocoinst ^. categories

    case image V.!? 0 of
      Nothing -> throwString "Invalid Image ID"
      Just image -> do
          img <- liftIO $ G.readImage (base </> split </> image ^. img_file_name)
          case img of
            Left msg -> throwString msg
            Right img -> do
                font_path <- liftIO $ G.findFontOfFamily "DejaVu Sans" (G.FontStyle False False) >>=
                                maybe (throwString "font not found") return
                font <- liftIO $ G.loadFontFile font_path >>= either throwString return

                let width     = image ^. img_width
                    height    = image ^. img_height
                    white     = G.PixelRGBA8 255 255 255 255
                    headColor = G.PixelRGBA8 0 0x86 0xc1 128
                    boxColor  = G.PixelRGBA8 0 0x86 0xc1 255
                    textColor = G.PixelRGBA8 255 0 0 255
                    pointSize = G.PointSize 8

                    buildAnno anno color = liftIO $ do
                        let cat_id = anno ^. ann_category_id
                            cat_name = M.lookupDefault "???" cat_id cat_table
                            bbox = anno ^. ann_bbox
                            COLOR.RGB r g b = COLOR.toSRGB24 color
                        mask <- getMask anno width height
                        return $ [
                            AnnoBoundingBox cat_name font pointSize textColor headColor bbox boxColor,
                            AnnoMask mask (G.PixelRGB8 r g b)]

                annotations <- fmap concat $ zipWithM buildAnno (V.toList annos) infiniteWebColors

                let rimg  = G.renderDrawing width height white $ do
                                G.drawImage (G.convertRGBA8 img) 0 (G.V2 0 0)
                                mapM_ renderAnnotation annotations
                    outfile = formatToString (int % ".png") imgid
                logInfo . display $ sformat ("Writing image: " % string) outfile
                liftIO $ G.writePng outfile rimg

                -- forM_ annos $ \anno -> liftIO $ do
                --     mask <- getMask anno width height
                --     let outfile = formatToString (int % "_" % int % ".png") imgid (anno ^. ann_id)
                --     G.writePng outfile mask

readFromCache path = do
  bs <- SBS.readFile path
  Store.decodeIO bs

readFromJson :: (MonadIO m, Aeson.FromJSON a) => FilePath -> m a
readFromJson path = do
  bs <- LBS.readFile path
  case Aeson.decode' bs of
    Nothing   -> error $ "cannot parse annotation file: " ++ path
    Just inst -> return inst

store path obj = do
  SBS.writeFile path (Store.encode obj)
  return obj

readAnnotations :: (Store.Store a, Aeson.FromJSON a)
                => FilePath -> IO a
readAnnotations path =
  readFromCache cache_file `catchIO`
  (\_ -> readFromJson path >>= store cache_file)
  where
    cache_file = "./instance.store"

getMask :: Annotation -> Int -> Int -> IO (G.Image G.Pixel8)
getMask anno width height = do
    crle <- case anno ^. ann_segmentation of
              SegRLE cnts _    -> frUncompressedRLE cnts height width
              SegPolygon (RNE.nonEmpty -> Just polys) ->
                  frPoly (RNE.map SV.fromList polys) height width
              _ -> throwString "Cannot build CRLE"
    crle <- merge crle False
    mask <- decode crle
    let Z :. c :. w :. h = RP.extent mask
    if (c > 1)
       then (throwString "More than 1 channel")
       else do
           -- HIP uses image HxW
           let image = RP.transpose $ RP.map (G.PixelY . (*255)) $ RP.reshape (Z :. w :. h) mask
           return $ G.toJPImageY8  $ G.exchange G.VS $ fromRepaArrayP image

data AnnoBuilder = AnnoBoundingBox
    { _anno_text      :: String
    , _anno_text_font :: Font
    , _anno_text_size :: PointSize
    , _anno_text_fg   :: G.PixelRGBA8
    , _anno_text_bg   :: G.PixelRGBA8
    , _anno_box       :: (Float, Float, Float, Float)
    , _anno_box_fg    :: G.PixelRGBA8
    }
    | AnnoMask
    { _anno_mask    :: G.Image G.Pixel8
    , _anno_mask_fg :: G.PixelRGB8
    }

renderAnnotation :: AnnoBuilder  -> G.Drawing G.PixelRGBA8 ()
renderAnnotation AnnoBoundingBox{..} = do
    let (x1, y1, w, h) = _anno_box
        BoundingBox tx1 ty1 tx2 ty2 _ = stringBoundingBox _anno_text_font 96 _anno_text_size _anno_text
        title = G.rectangle (G.V2 (x1-1) (y1-ty2+ty1-1)) (tx2-tx1+2) (ty2-ty1+1)
        textBase = G.V2 x1 y1
        rect = G.rectangle (G.V2 x1 y1) w h
    G.withTexture (G.uniformTexture _anno_text_bg) $ do
        G.fill title
    G.withTexture (G.uniformTexture _anno_text_fg) $ do
        G.printTextAt _anno_text_font _anno_text_size (G.V2 x1 (y1-2)) _anno_text
    G.withTexture (G.uniformTexture _anno_box_fg) $ do
        G.stroke 1 G.JoinRound (G.CapRound, G.CapRound) rect

renderAnnotation AnnoMask{..} = do
    let G.PixelRGB8 r g b = _anno_mask_fg
        color_mask = G.pixelMap (\m -> G.PixelRGBA8 r g b (floor $ fromIntegral m * 0.6)) _anno_mask
    G.drawImage color_mask 0 (G.V2 0 0)
