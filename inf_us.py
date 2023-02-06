# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.14.0
#   kernelspec:
#     display_name: Python 3 (ipykernel)
#     language: python
#     name: python3
# ---

# %% [markdown]
# # **Running H2OvalNet on Cloud-Optimized GeoTIFFs**
#
# [![Open on Planetary Computer](https://img.shields.io/badge/-Open%20on%20Planetary%20Computer-blue)](https://pccompute.westeurope.cloudapp.azure.com/compute/hub/user-redirect/git-pull?repo=https%3A%2F%2Fcode.osu.edu%2Fbyrdhowatgroup%2Fh2oval&urlpath=lab%2Ftree%2Fh2oval%2Fh2oval_cloud_inference.ipynb&branch=main)
#
# Scalable inference for the entire planet!
# Loading Sentinel-2 and Copernicus DEM data
# and running machine learning inference in the cloud.
#
# To use this notebook fully, note that you will need a trained neural network
# model file located in the same folder where this script is being run:
# - h2ovalnet.onnx (trained model in Open Neural Network Exchange format)
#
# There is also one (optional) environment variables to set:
# - SAS_TOKEN (required to upload files to Azure Blob Storage)

# %%
import argparse
import gc
import os
import sys
import warnings
from typing import Any, Callable, Dict, List, Optional, cast

import azure.storage.blob
import geopandas as gpd
import numpy as np
import onnxruntime
import pandas as pd
import planetary_computer
import pystac_client
import rasterio
import rioxarray
import shapely.geometry
import shapely.ops
import stackstac
import torch
import torchgeo.samplers
import tqdm
import xarray as xr
from rasterio.crs import CRS
from rtree.index import Index, Property
from torchgeo.datasets.geo import GeoDataset
from torchgeo.datasets.utils import BoundingBox

# https://github.com/geopandas/geopandas/issues/2347
warnings.filterwarnings("ignore", message="pandas.Float64Index")
warnings.filterwarnings("ignore", message="pandas.Int64Index")

# %% [markdown]
# ## Find Sentinel-2 and Copernicus DEM Cloud-Optimized GeoTIFFs
#
# Search for 12 of the best cloud-free Sentinel-2 images in a year,
# and get the corresponding Copernicus DEM over the same spatial region.

# %%
# Define area of interest
# area_of_interest = shapely.geometry.box(minx=112.0, miny=-44.0, maxx=154.0, maxy=-10.0)
area_of_interest = shapely.geometry.box(minx=-125.0, miny=25.2, maxx=-66.0, maxy=49.0)

# Define temporal range
daterange: dict = {"interval": ["2021-04-01T00:00:00Z", "2022-03-31T23:59:59Z"]}

# %%
# Set tile id to process
parser = argparse.ArgumentParser()
parser.add_argument(
    "--mgrs-tile", default="53HQB", help="Military grid reference system tile ID"
)

try:
    args = parser.parse_args()
except SystemExit:
    args = parser.parse_args(args=[])
print(f"Processing MGRS tile: {args.mgrs_tile}")

# %% [markdown]
# Using `pystac_client` we can search the Planetary Computer's
# STAC endpoint for items matching our query parameters.
#
# Full list of datasets at https://planetarycomputer.microsoft.com/catalog

# %%
stac = pystac_client.Client.open(
    url="https://planetarycomputer.microsoft.com/api/stac/v1"
)

# %%
# Define search with CQL2 syntax
# Things like bounding box, dataset collection, cloud cover percentage, etc
sen2_search: pystac_client.item_search.ItemSearch = stac.search(
    filter_lang="cql2-json",
    filter={
        "op": "and",
        "args": [
            {
                "op": "s_intersects",
                "args": [{"property": "geometry"}, area_of_interest.__geo_interface__],
            },
            {"op": "anyinteracts", "args": [{"property": "datetime"}, daterange]},
            {"op": "=", "args": [{"property": "collection"}, "sentinel-2-l2a"]},
            {
                "op": "like",
                "args": [{"property": "s2:mgrs_tile"}, f"%{args.mgrs_tile}%"],
            },
            {"op": "<=", "args": [{"property": "eo:cloud_cover"}, 25]},
            {"op": "<=", "args": [{"property": "s2:nodata_pixel_percentage"}, 25]},
        ],
    },
)
sen2_items = list(sen2_search.items())
assert len(sen2_items) > 0  # need to have at least 1 Sentinel-2 image in the stack!

# Get Sentinel-2 image's bounding box extent
geoms = [
    shapely.geometry.box(*item.bbox) for item in sen2_items if item.bbox is not None
]
sen2_aoi = shapely.geometry.box(*shapely.ops.unary_union(geoms=geoms).bounds)

# Get corresponding Copernicus DEM
dem_search: pystac_client.item_search.ItemSearch = stac.search(
    filter_lang="cql2-json",
    filter={
        "op": "and",
        "args": [
            {
                "op": "s_intersects",
                "args": [{"property": "geometry"}, sen2_aoi.__geo_interface__],
            },
            {
                "op": "=",
                "args": [{"property": "collection"}, "cop-dem-glo-30"],
            },
        ],
    },
)
dem_items = list(dem_search.items())

print(f"{len(sen2_items)} Sentinel-2 items and {len(dem_items)} Copernicus DEM items")

# %%
# Get 12 images with the least cloud cover and least NaN pixels
df = pd.DataFrame(data=[s.properties for s in sen2_items])
top12_cloud_cover = df.sort_values(
    by=["eo:cloud_cover", "s2:nodata_pixel_percentage"], ascending=True
).head(12)
print(
    f"Using only 12 images with <= {float(top12_cloud_cover.tail(n=1)['eo:cloud_cover'])} cloud cover"
)
print(top12_cloud_cover)
_sen2_items: list = [sen2_items[i] for i in top12_cloud_cover.index]
assert len(_sen2_items) == 12

# %% [markdown]
# ## Process Sentinel-2 time series and Copernicus DEM
#
# Creating the spatiotemporal Sentinel-2 image stack,
# and reprojecting the Copernicus DEM to match.

# %%
signed_sen2_items: list = [
    planetary_computer.sign(item).to_dict() for item in _sen2_items
]
signed_dem_items: list = [planetary_computer.sign(item).to_dict() for item in dem_items]

# %%
# Sentinel-2 image stack
da_sen2: xr.DataArray = (
    stackstac.stack(
        signed_sen2_items,
        assets=[
            "B02",  # Blue ~493nm 10m
            "B03",  # Green ~560nm 10m
            "B04",  # Red ~665nm 10m
            "B05",  # VNIR ~ 704nm 20m
            "B06",  # VNIR ~740nm 20m
            "B07",  # VNIR ~783nm 20m
            "B08",  # NIR ~833nm 10m
            "B8A",  # NIR ~865nm 20m
            "B11",  # SWIR ~1610nm 20m
            "B12",  # SWIR ~2190nm 20m
        ],
        chunksize=3660,  # https://github.com/microsoft/PlanetaryComputer/discussions/17#discussioncomment-2518045
        resolution=10,  # 10 metres
        rescale=False,
        dtype=np.float16,
        fill_value=np.nan,
    )
    # .where(lambda x: x > 0, other=np.nan)  # sentinel-2 uses 0 as nodata
    .assign_coords(band=lambda x: x.common_name.rename("band"))  # use common names
)
print(da_sen2)

# %%
# Copernicus DEM image stack
da_dem: xr.DataArray = stackstac.stack(
    signed_dem_items,
    assets=["data"],
    # chunksize=3660,
    # resolution=10,  # 30 -> 10 metres
    # rescale=False,
    # dtype=np.float16,
    # fill_value=np.nan,
)
print(da_dem)

# %%
# Naive way of mosaicking the DEM image stack
_dem = da_dem.max(dim="time").astype(dtype=np.int16)
_dem = _dem.persist()

# %%
# %%time
# Reproject the DEM in EPSG:4326 to match the Sentinel-2 image in UTM
# TODO reproject on cluster using dask instead of running 'locally', see
# https://github.com/corteva/rioxarray/issues/119
dem_crs = rasterio.crs.CRS.from_epsg(code=int(4326))
_ = _dem.rio.set_crs(input_crs=dem_crs)
_dem = _dem.rio.reproject_match(
    match_data_array=da_sen2, resampling=rasterio.enums.Resampling.bilinear
)
dem = _dem.rio.clip_box(*da_sen2.rio.bounds())
dem

# %%
dem.rio.crs, dem.rio.resolution()

# %%
# dem.isel(band=0).plot.imshow()

# %% [raw]
# dem.rio.to_raster(
#     raster_path=f"Copernicus_DSM_COG_{args.mgrs_tile}.tif",
#     driver="COG",
#     dtype="int16",
#     compress="zstd",
# )

# %%
# Edit DEM DataArray metadata so that it can be
# concatenated with the Sentinel-2 image
dem["band"] = ["dem"]
dem["epsg"] = da_sen2.epsg
dem["title"] = "DEM - 30m"
dem["common_name"] = "dem"
dem["center_wavelength"] = "none"
dem["full_width_half_max"] = 0  # ??
dem["platform"] = "TanDEM-X"
# for t in tqdm.trange(0, 1):  # TODO loop through Sentinel-2 temporal images properly
#     img = xr.concat(
#         objs=[da_sen2.isel(time=t).drop_vars(names="platform"), dem], dim="band"
#     ).astype(dtype=np.int16)
# img

# %% [raw]
# # TODO maybe store Sentinel-2 as uint16
# # and Copernicus DEM as int16, otherwise almost 2GB per file...
# img.rio.to_raster(
#     raster_path=f"11_band_COG_{args.mgrs_tile}.tif",
#     driver="COG",
#     # technically should save as int16 since DEM can be <0
#     # but use uint16 for sake of disk space
#     dtype="uint16",
#     compress="zstd",
# )

# %%

# %% [markdown]
# torchgeo.XarrayGeoDataset, https://github.com/microsoft/torchgeo/pull/509


# %%
class RioXarrayDataset(GeoDataset):
    """Wrapper for geographical datasets stored as an xarray.DataArray.

    Relies on rioxarray.
    """

    def __init__(
        self,
        xr_dataarray: xr.DataArray,
        crs: Optional[CRS] = None,
        res: Optional[float] = None,
        transforms: Optional[Callable[[Dict[str, Any]], Dict[str, Any]]] = None,
    ) -> None:
        """Initialize a new Dataset instance.

        Args:
            xr_dataarray: n-dimensional xarray.DataArray
            crs: :term:`coordinate reference system (CRS)` to warp to
                (defaults to the CRS of dataarray)
            res: resolution of the dataset in units of CRS
                (defaults to the resolution of the dataarray)
            transforms: a function/transform that takes an input sample
                and returns a transformed version
        """
        super().__init__(transforms)

        self.xr_dataarray = xr_dataarray
        self.transforms = transforms

        # Create an R-tree to index the dataset
        self.index = Index(interleaved=False, properties=Property(dimension=3))

        # Populate the dataset index
        if crs is None:
            crs = xr_dataarray.rio.crs
        if res is None:
            res = xr_dataarray.rio.resolution()[0]

        (minx, miny, maxx, maxy) = xr_dataarray.rio.bounds()
        if hasattr(xr_dataarray, "time"):
            mint = int(xr_dataarray.time.min().data)
            maxt = int(xr_dataarray.time.max().data)
        else:
            mint = 0
            maxt = sys.maxsize
        coords = (minx, maxx, miny, maxy, mint, maxt)
        self.index.insert(0, coords, xr_dataarray.name)

        self._crs = cast(CRS, crs)
        self.res = cast(float, res)

    def __getitem__(self, query: BoundingBox) -> Dict[str, Any]:
        """Retrieve image/mask and metadata indexed by query.

        Args:
            query: (minx, maxx, miny, maxy, mint, maxt) coordinates to index

        Returns:
            sample of image/mask and metadata at that index

        Raises:
            IndexError: if query is not found in the index
        """
        hits = self.index.intersection(tuple(query), objects=True)
        items = [hit.object for hit in hits]

        if not items:
            raise IndexError(
                f"query: {query} not found in index with bounds: {self.bounds}"
            )

        image = self.xr_dataarray.rio.clip_box(
            minx=query.minx, miny=query.miny, maxx=query.maxx, maxy=query.maxy
        )
        sample = {"image": torch.tensor(image.data), "crs": self.crs, "bbox": query}

        if self.transforms is not None:
            sample = self.transforms(sample)

        return sample


# %% [markdown]
# ## Run inference using ONNX model
#
# Generate bounding box predictions!
# Note that you will need to have the
# trained neural network model file (`h2ovalnet.onnx`)
# present on the local instance.
#
# References:
# - https://pytorch-lightning.readthedocs.io/en/1.6.0/common/production_inference.html#convert-to-onnx
# - https://onnxruntime.ai/docs/api/python/api_summary.html#load-and-run-a-model

# %%
print(onnxruntime.__version__)
print(onnxruntime.get_available_providers())

# %%
# Setup ONNX Runtime session
ort_session = onnxruntime.InferenceSession(
    path_or_bytes="h2ovalnet.onnx",
    providers=["CUDAExecutionProvider", "CPUExecutionProvider"],
)
input_name = ort_session.get_inputs()[0].name
print(input_name)


# %%
def get_geo_boxes(
    bbox_outputs: list, crses: list, extents: list, uri: str
) -> gpd.GeoDataFrame:
    """
    Given predicted bounding box tensors from neural network that
    are in image coordinates, georeference the boxes using affine
    transform based on supplied coordinate reference system and
    image extent.
    """
    geo_boxes: list = []

    # Get bounding box and scores, but not the labels.
    # Since this is a 1-class problem, we can just
    # get the first item (shape (22, 5)) in each consecutive pair,
    # and ignore the labels (shape (22,)) which would be all '0'.
    bbox_and_scores = [bbox_outputs[idx] for idx in range(0, len(bbox_outputs), 2)]
    assert len(bbox_and_scores) == 16

    for idx, bbox_and_score in enumerate(bbox_and_scores):
        boxes: torch.Tensor = bbox_and_score[:, :4]
        scores: torch.Tensor = bbox_and_score[:, 4]

        # extent = rasterio.coords.BoundingBox(
        #     left=_da.x - 2400, bottom=_da.y - 2400, right=_da.x + 2400, top=_da.y + 2400
        # )
        extent: rasterio.coords.BoundingBox = extents[idx]

        # Georeference predicted bounding boxes by converting from image
        # coordinates to geographical coordinates using affine transform
        _gdf = gpd.GeoDataFrame(
            # data={"score": scores, "isotime": _da.time.data},
            data={
                "score": scores,
                "isotime": pd.Timestamp(extent.mint),
                # Add input Sentinel-2 product URI
                # TODO, include Copernicus DEM URI too?
                "input_img_uri": uri,
            },
            geometry=[
                shapely.affinity.affine_transform(
                    geom=shapely.geometry.box(*coords),
                    matrix=[10, 0, 0, -10, extent.minx, extent.maxy],
                    # matrix=[10, 0, 0, -10, extent.left, extent.top],
                )
                for coords in boxes
            ],
            crs=crses[idx],
        )
        # _gdf["input_img_uri"] = _gdf.input_img_uri.astype(dtype="string[pyarrow]")
        geo_boxes.append(_gdf.to_crs(crs="EPSG:4326"))

    # Gather all the box polygons in a batch into a single GeoDataFrame
    geodataframe: gpd.GeoDataFrame = pd.concat(objs=geo_boxes)
    geodataframe.set_crs(crs="EPSG:4326", inplace=True)

    return geodataframe


# %%
# %%time
b: int = 0
# Temporal loop through Sentinel-2 time-series
for t in tqdm.auto.trange(0, len(da_sen2.time), desc=f"MGRS {args.mgrs_tile}"):
    # Get 11-band image for one spatiotemporal slice
    stac_image: xr.DataArray = xr.concat(
        objs=[da_sen2.isel(time=t).drop_vars(names="platform"), dem], dim="band"
    ).astype(dtype=np.int16)
    uri: str = str(stac_image["s2:product_uri"].data)

    # Spatial sliding window loop to run inference on 960x960 image chips
    sampler = torchgeo.samplers.GridGeoSampler(
        dataset := RioXarrayDataset(xr_dataarray=stac_image.compute()),
        size=960,
        stride=960,
    )

    for batch in tqdm.auto.tqdm(
        iterable=torch.utils.data.DataLoader(
            dataset=dataset,
            batch_size=16,
            sampler=sampler,
            collate_fn=torchgeo.datasets.stack_samples,
            drop_last=False,
        ),
        desc=f"t={t}:{uri[:44]}",
    ):
        # Get a tensor of shape (16, 11, 960, 960)
        # Note that tensor needs to be converted from int16 to float32
        images: np.ndarray = (batch["image"] / 2**16).to(dtype=torch.float32).numpy()
        assert np.all(a=~np.isnan(images))  # Ensure no NaNs

        # Pass the tensor into the ONNX network to get predictions!
        ort_inputs = {input_name: images}  # .clip(min=0) ensure minimum is 0
        try:
            run_options = onnxruntime.RunOptions()
            run_options.log_severity_level = 4  # log Fatal errors only
            ort_outs = ort_session.run(
                output_names=None, input_feed=ort_inputs, run_options=run_options
            )
            del images  # free up memory
        except onnxruntime.capi.onnxruntime_pybind11_state.RuntimeException as e:
            continue

        # Parse predicted outputs
        # The first 10 tensors (index 0-9) are not needed
        # as they are just intermediate outputs
        bbox_outputs = [o for o in ort_outs[9:]]
        # The other tensors are like [(22, 5), (22,), (21, 5), (21,), ...]
        # I.e., consecutive pairs of bbox_and_score and labels
        assert all(
            bbox_outputs[idx].shape[0] == bbox_outputs[idx + 1].shape[0]
            for idx in range(0, len(bbox_outputs), 2)
        )

        # Get georeferenced bounding boxes
        crses: List[rasterio.crs.CRS] = batch["crs"]  # coord reference system
        extents: List[rasterio.coords.BoundingBox] = batch["bbox"]  # img extent
        _geodataframe: gpd.GeoDataFrame = get_geo_boxes(
            bbox_outputs=bbox_outputs, crses=crses, extents=extents, uri=uri
        )

        # Write/append bounding boxes from 1 time slice to GeoPackage
        _geodataframe.to_file(
            filename := f"pred_boxes_{args.mgrs_tile}.gpkg",
            driver="GPKG",
            mode="w" if b == 0 else "a",  # write first, append later
        )
        b += 1

        # Free up memory
        del _geodataframe
        gc.collect()

    # Free up memory again
    del dataset
    gc.collect()
    # break

# %%
# Convert GeoPackage to FlatGeoBuf
geodataframe: gpd.GeoDataFrame = gpd.read_file(filename=filename)
geodataframe.to_file(
    filename := f"pred_boxes_{args.mgrs_tile}.fgb", driver="FlatGeobuf"
)
print(f"Saved {len(geodataframe)} polygons to {filename}")

# %%
# Upload bounding box polygons FlatGeobuf to Azure Blob Storage
# https://planetarycomputer.microsoft.com/docs/quickstarts/storage/#Write-to-Azure-Blob-Storage
# https://kbatch.readthedocs.io/en/latest/examples/ndvi-blob-storage.html
if credential := os.getenv("SAS_TOKEN"):
    container_client = azure.storage.blob.ContainerClient(
        account_url := "https://moortgat.blob.core.windows.net",
        container_name="kbatch",
        credential=credential,
    )
    with open(file=filename, mode="rb") as f:
        blob_obj: azure.storage.blob.BlobClient = container_client.upload_blob(
            name=f"data_US/predictions/{filename}", data=f, overwrite=True
        )
    print(f"Uploaded to {account_url}/{blob_obj.blob_name}")
