{-# LANGUAGE OverloadedStrings #-}
{-# LANGUAGE TemplateHaskell #-}
module MXNet.Coco.Types where

import RIO hiding (Category)
import RIO.Char (ord)
import RIO.Time (LocalTime(..), TimeOfDay(..), Day, fromGregorianValid)
import Data.Aeson
import qualified Data.Attoparsec.Text as A
import Data.Bits ((.&.))
import Control.Lens (makeLenses)
import GHC.Generics (Generic)
import Data.Store (Store)

data Instance = Instance {
    _info :: Info,
    _images :: Vector Image,
    _annotations :: Vector Annotation,
    _licenses :: Vector License,
    _categories :: Vector Category
} deriving Generic

instance Store Instance
instance NFData Instance

instance FromJSON Instance where
    parseJSON = withObject "Instance" $ \v -> Instance
        <$> v .: "info"
        <*> v .: "images"
        <*> v .: "annotations"
        <*> v .: "licenses"
        <*> v .: "categories"

data Info = Info {
    _info_year :: Int,
    _info_version :: String,
    _info_description :: String,
    _info_contributor :: String,
    _info_url :: String,
    _info_date_created :: CocoDay
} deriving Generic

instance Store Info
instance NFData Info

instance FromJSON Info where
    parseJSON = withObject "Info" $ \v -> Info
        <$> v .: "year"
        <*> v .: "version"
        <*> v .: "description"
        <*> v .: "contributor"
        <*> v .: "url"
        <*> v .: "date_created"

data License = License {
    _lic_id :: Int,
    _lic_name :: String,
    _lic_url :: String
} deriving Generic

instance Store License
instance NFData License

instance FromJSON License where
    parseJSON = withObject "License" $ \v -> License
        <$> v .: "id"
        <*> v .: "name"
        <*> v .: "url"

data Image = Image {
    _img_id :: !Int,
    _img_width :: !Int,
    _img_height :: !Int,
    _img_file_name :: !String,
    _img_license :: !Int,
    _img_flickr_url :: !String,
    _img_coco_url :: !String,
    _img_date_captured :: !LocalTime
} deriving (Generic, Show)

deriving instance Generic TimeOfDay
deriving instance Generic LocalTime
instance Store TimeOfDay
instance Store LocalTime
instance Store Image
instance NFData Image

instance FromJSON Image where
    parseJSON = withObject "Image" $ \v -> Image
        <$> v .: "id"
        <*> v .: "width"
        <*> v .: "height"
        <*> v .: "file_name"
        <*> v .: "license"
        <*> v .: "flickr_url"
        <*> v .: "coco_url"
        <*> v .: "date_captured"

data Annotation = AnnObjectDetection {
    _ann_id :: !Int,
    _ann_image_id :: !Int,
    _ann_category_id :: !Int,
    _ann_segmentation :: !Segmentation,
    _ann_area :: !Float,
    _ann_bbox :: !(Float, Float, Float, Float)
} deriving Generic

instance Store Annotation
instance NFData Annotation

instance FromJSON Annotation where
    parseJSON = withObject "Annotation" $ \v -> AnnObjectDetection
        <$> v .: "id"
        <*> v .: "image_id"
        <*> v .: "category_id"
        <*> v .: "segmentation"
        <*> v .: "area"
        <*> v .: "bbox"

data Segmentation = SegRLE { _seg_counts :: [Int], _seg_size :: (Int, Int)} | SegPolygon [[Double]]
  deriving Generic

instance Store Segmentation
instance NFData Segmentation

instance FromJSON Segmentation where
    parseJSON value = (withObject "RLE" (\v -> SegRLE <$> v .: "counts" <*> v .: "size") value) <|>
                      (withArray "Polygon" (\v -> SegPolygon <$> parseJSONList (Array v)) value)
data Category = CatObjectDetection {
    _odc_id :: Int,
    _odc_name :: String,
    _odc_supercategory :: String
} deriving Generic

instance Store Category
instance NFData Category

instance FromJSON Category where
    parseJSON = withObject "Category" $ \v -> CatObjectDetection
        <$> v .: "id"
        <*> v .: "name"
        <*> v .: "supercategory"

newtype CocoDay = CocoDay Day deriving Generic

instance Store CocoDay
instance NFData CocoDay

instance FromJSON CocoDay where
    parseJSON = withText "Day" $ \t -> case A.parseOnly (day <* A.endOfInput) t of
        Left err -> fail $ "could not parse date: " ++ err
        Right r  -> return r
      where
        day = do
            y <- (A.decimal <* A.char '/') <|> fail "date must be of form YYYY/MM/DD"
            m <- (twoDigits <* A.char '/') <|> fail "date must be of form YYYY/MM/DD"
            d <- twoDigits <|> fail "date must be of form YYYY/MM/DD"
            maybe (fail "invalid date") return (CocoDay <$> fromGregorianValid y m d)
        twoDigits = do
              a <- A.digit
              b <- A.digit
              let c2d c = ord c .&. 15
              return $! c2d a * 10 + c2d b

makeLenses ''Instance
makeLenses ''Info
makeLenses ''License
makeLenses ''Image
makeLenses ''Annotation
makeLenses ''Segmentation
makeLenses ''Category
