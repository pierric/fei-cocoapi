module MXNet.Coco.Index where

import RIO hiding (Category)
import qualified Data.Vector as V (filter, null, head)

import MXNet.Coco.Types

allCats :: Instance -> Vector Category
allCats = (^. categories)

allAnns :: Instance -> Vector Annotation
allAnns = (^. annotations)

catByName :: String -> Vector Category -> Maybe Category
catByName name = vecToMaybe . V.filter (\cat -> cat ^. odc_name == name)

annsByCat :: Category -> Vector Annotation -> Vector Annotation
annsByCat cat = V.filter (\ann -> ann ^. ann_category_id == cat ^. odc_id )

annsByImg :: Image -> Vector Annotation -> Vector Annotation
annsByImg img = V.filter (\ann -> ann ^. ann_image_id == img ^. img_id )

annByCatImg :: Image -> Category -> Vector Annotation -> Maybe Annotation
annByCatImg img cat = vecToMaybe . annsByCat cat . annsByImg img

vecToMaybe :: Vector a -> Maybe a
vecToMaybe vec | V.null vec = Nothing
               | otherwise = Just $ V.head vec
