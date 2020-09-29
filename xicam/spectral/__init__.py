import numpy as np
from qtpy.QtWidgets import QLabel, QComboBox, QHBoxLayout, QWidget, QSpacerItem, QSizePolicy
from xicam.core import msg
from xicam.plugins import GUIPlugin, GUILayout
from xicam.plugins.guiplugin import PanelState
from xicam.gui.widgets.imageviewmixins import XArrayView, CatalogView#, DepthPlot, BetterTicks, BetterLayout, BetterPlots
import logging
#from xicam.gui.widgets.library import LibraryWidget
from xicam.gui.widgets.linearworkfloweditor import WorkflowEditor
from databroker.core import BlueskyRun
from xicam.core.execution import Workflow
from xarray import DataArray


def project_nxSTXM(run_catalog: BlueskyRun):
    _, projection = next(filter(lambda projection: projection[0] == 'nxSTXM', run_catalog.metadata['start']['projections']))
    stream, field = projection['irmap/DATA/data']
    sample_x = projection['irmap/DATA/sample_x']
    sample_y = projection['irmap/DATA/sample_y']
    energy = projection['irmap/DATA/energy']

    xdata = getattr(run_catalog, stream).to_dask()[field]  # type: DataArray

    xdata = np.squeeze(xdata)

    xdata = xdata.assign_coords({xdata.dims[0]: energy, xdata.dims[1]: sample_y, xdata.dims[2]: sample_x})

    return xdata

def project_NXcxi_ptycho(run_catalog: BlueskyRun):
    projection = next(filter(lambda projection: projection['name'] == 'nxCXI_ptycho', run_catalog.metadata['start']['projections']))

    rec_stream = projection['projection']['data']['stream']
    rec_field = projection['projection']['data']['field']
    energy_stream = projection['projection']['energy']['stream']
    energy_field = projection['projection']['energy']['field']
    coords_x_stream = projection['projection']['coords_x']['stream']
    coords_x_field = projection['projection']['coords_x']['field']
    coords_y_stream = projection['projection']['coords_y']['stream']
    coords_y_field = projection['projection']['coords_y']['field']

    rec_data = getattr(run_catalog, rec_stream).to_dask()

# class CatalogViewerBlend(BetterPlots, BetterLayout, DepthPlot, XArrayView):
#     def __init__(self, *args, **kwargs):
#         # CatalogViewerBlend inherits methods from XArrayView and CatalogView
#         # super allows us to access both methods when calling super() from Blend
#         super(CatalogViewerBlend, self).__init__(*args, **kwargs)


class SpectralPlugin(GUIPlugin):
    name = "Spectral"

    def __init__(self):
        # self.catalog_viewer = CatalogViewerBlend()
        self.catalog_viewer = CatalogView()
        #self.library_viewer = LibraryWidget()

        self.treatment_workflow = Workflow()

        self.stages = {
            "Acquire": GUILayout(QWidget()),
            #"Library": GUILayout(left=PanelState.Disabled, lefttop=PanelState.Disabled, center=self.library_viewer, right=self.catalog_viewer),
            "Map": GUILayout(self.catalog_viewer, right=WorkflowEditor(self.treatment_workflow)),
            "Decomposition": GUILayout(QWidget()),
            "Clustering": GUILayout(QWidget()),
        }
        super(SpectralPlugin, self).__init__()

    def appendCatalog(self, run_catalog, **kwargs):
        try:
            self.stream_fields = get_all_image_fields(run_catalog)
            stream_names = get_all_streams(run_catalog)

            msg.showMessage(f"Loading primary image for {run_catalog.name}")
            # # try and startup with primary catalog and whatever fields it has
            # if "primary" in self.stream_fields:
            #     default_stream_name = "primary" if "primary" in stream_names else stream_names[0]
            # else:
            #     default_stream_name = list(self.stream_fields.keys())[0]

            # Apply nxSTXM projection
            xdata = project_nxSTXM(run_catalog)

            self.catalog_viewer.setImage(xdata)

        except Exception as e:
            msg.logError(e)
            msg.showMessage("Unable to display: ", str(e))

    def run_workflow(self):
        """Run the internal workflow.
        In this example, this will be called whenever the "Run Workflow" in the WorkflowEditor is clicked.
        """
        if not self.catalog_viewer.catalog:  # Don't run if there is no data loaded in
            return
        # Use Workflow's execute method to run the workflow.
        # our callback_slot will be called when the workflow has executed its operations
        # image is an additional keyword-argument that is fed into the first operation in the workflow
        # (the invert operation needs an "image" argument)
        self._workflow.execute(callback_slot=self.results_ready,
                               image=self.catalog_viewer.image)

    def results_ready(self, *results):
        """Update the results view widget with the processed data.
        This is called when the workflow's execute method has finished running is operations.
        """
        # print(results)
        # results is a tuple that will look like:
        # ({"output_name": output_value"}, ...)
        # This will only contain more than one dictionary if using Workflow.execute_all
        output_image = results[0]["output_image"]  # We want the output_image from the last operation
        self.results_viewer.setImage(output_image)  # Update the result view widget

### small helper functions
def get_stream_data_keys(run_catalog, stream):
    return run_catalog[stream].metadata["descriptors"][0]["data_keys"]


def get_all_streams(run_catalog):
    return list(run_catalog)


def get_all_image_fields(run_catalog):
    all_streams_image_fields = {}
    for stream in get_all_streams(run_catalog):
        stream_fields = get_stream_data_keys(run_catalog, stream)
        field_names = stream_fields.keys()
        for field_name in field_names:
            field_shape = len(stream_fields[field_name]["shape"])
            if field_shape > 1 and field_shape < 5:
                # if field contains at least 1 entry that is at least one-dimensional (shape=2)
                # or 2-dimensional (shape=3) or up to 3-dimensional (shape=4)
                # then add field e.g. 'fccd_image'
                if stream in all_streams_image_fields.keys():  # add values to stream dict key
                    all_streams_image_fields[stream].append(field_name)
                else:  # if stream does not already exist in dict -> create new entry
                    all_streams_image_fields[stream] = [field_name]
            # TODO how to treat non image data fields in streams
            # else:
    return all_streams_image_fields

