module MXNet.Coco.Index where

import Control.Lens ((^.))
import qualified Data.Vector as V (Vector, filter, null, head)

import MXNet.Coco.Types

allCats :: Instance -> V.Vector Category
allCats = (^. categories)

allAnns :: Instance -> V.Vector Annotation
allAnns = (^. annotations)

catByName :: String -> V.Vector Category -> Maybe Category
catByName name = vecToMaybe . V.filter (\cat -> cat ^. odc_name == name)

annsByCat :: Category -> V.Vector Annotation -> V.Vector Annotation
annsByCat cat = V.filter (\ann -> ann ^. ann_category_id == cat ^. odc_id )

annsByImg :: Image -> V.Vector Annotation -> V.Vector Annotation
annsByImg img = V.filter (\ann -> ann ^. ann_image_id == img ^. img_id )

annByCatImg :: Image -> Category -> V.Vector Annotation -> Maybe Annotation
annByCatImg img cat = vecToMaybe . annsByCat cat . annsByImg img

vecToMaybe :: V.Vector a -> Maybe a
vecToMaybe vec | V.null vec = Nothing
               | otherwise = Just $ V.head vec