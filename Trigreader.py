import numpy as np
import matplotlib.pyplot as plt

import astropy.io.fits as fits

lu = (
    "n0",
    "n1",
    "n2",
    "n3",
    "n4",
    "n5",
    "n6",
    "n7",
    "n8",
    "n9",
    "na",
    "nb",
    "b0",
    "b1",
)


class TrigReader(object):
    """
    This class reads a GBM trigdat file and performs background fitting, source selection, and plotting.
    It is also fed to the Balrog when performing localization with trigdat data.
    :param triddat_file: string that is the path to the trigdat file you wish ot read
    :param fine: optional argument to use trigdat fine resolution data. Defaults to False
    :poly_order: optional argument to set the order of the polynomial used in the background fit.
    """

    def __init__(
        self,
        trigdat_file,
        fine=False,
        time_resolved=False,
        verbose=True,
        poly_order=-1,
        restore_poly_fit=None,
    ):

        # self._backgroundexists = False
        # self._sourceexists = False

        self._verbose = verbose
        self._time_resolved = time_resolved
        self._poly_order = poly_order
        self._restore_poly_fit = restore_poly_fit
        # Read the trig data file and get the appropriate info

        trigdat = fits.open(trigdat_file)
        self._filename = trigdat_file
        self._out_edge_bgo = np.array(
            [150.0, 400.0, 850.0, 1500.0, 3000.0, 5500.0, 10000.0, 20000.0, 50000.0],
            dtype=np.float32,
        )
        self._out_edge_nai = np.array(
            [3.4, 10.0, 22.0, 44.0, 95.0, 300.0, 500.0, 800.0, 2000.0], dtype=np.float32
        )
        self._binwidth_bgo = self._out_edge_bgo[1:] - self._out_edge_bgo[:-1]
        self._binwidth_nai = self._out_edge_nai[1:] - self._out_edge_nai[:-1]

        # Get the times
        evntrate = "EVNTRATE"

        self._trigtime = trigdat[evntrate].header["TRIGTIME"]
        self._tstart = trigdat[evntrate].data["TIME"] - self._trigtime
        self._tstop = trigdat[evntrate].data["ENDTIME"] - self._trigtime

        self._rates = trigdat[evntrate].data["RATE"]

        num_times = len(self._tstart)
        self._rates = self._rates.reshape(num_times, 14, 8)

        # Obtain the positional information
        self._qauts = trigdat[evntrate].data["SCATTITD"]  # [condition][0]
        self._sc_pos = trigdat[evntrate].data["EIC"]  # [condition][0]

        # Get the flight software location
        self._fsw_ra = trigdat["PRIMARY"].header["RA_OBJ"]
        self._fsw_dec = trigdat["PRIMARY"].header["DEC_OBJ"]
        self._fsw_err = trigdat["PRIMARY"].header["ERR_RAD"]

        # Clean up
        trigdat.close()

        # Sort out the high res times because they are dispersed with the normal
        # times.

        # The delta time in the file.
        # This routine is modeled off the procedure in RMFIT.
        myDelta = self._tstop - self._tstart
        self._tstart[myDelta < 0.1] = np.round(self._tstart[myDelta < 0.1], 4)
        self._tstop[myDelta < 0.1] = np.round(self._tstop[myDelta < 0.1], 4)

        self._tstart[~(myDelta < 0.1)] = np.round(self._tstart[~(myDelta < 0.1)], 3)
        self._tstop[~(myDelta < 0.1)] = np.round(self._tstop[~(myDelta < 0.1)], 3)

        if fine:

            # Create a starting list of array indices.
            # We will dumb then ones that are not needed

            all_index = list(range(len(self._tstart)))

            # masks for all the different delta times and
            # the mid points for the different binnings
            temp1 = myDelta < 0.1
            temp2 = np.logical_and(myDelta > 0.1, myDelta < 1.0)
            temp3 = np.logical_and(myDelta > 1.0, myDelta < 2.0)
            temp4 = myDelta > 2.0
            midT1 = (self._tstart[temp1] + self._tstop[temp1]) / 2.0
            midT2 = (self._tstart[temp2] + self._tstop[temp2]) / 2.0
            midT3 = (self._tstart[temp3] + self._tstop[temp3]) / 2.0

            # Dump any index that occurs in a lower resolution
            # binning when a finer resolution covers the interval

            for indx in np.where(temp2)[0]:
                for x in midT1:
                    if self._tstart[indx] < x < self._tstop[indx]:
                        if indx in all_index:
                            all_index.remove(indx)

            for indx in np.where(temp3)[0]:
                for x in midT2:
                    if self._tstart[indx] < x < self._tstop[indx]:
                        if indx in all_index:
                            all_index.remove(indx)

            for indx in np.where(temp4)[0]:
                for x in midT3:
                    if self._tstart[indx] < x < self._tstop[indx]:
                        if indx in all_index:
                            all_index.remove(indx)

            all_index = np.array(all_index)
        else:

            # Just deal with the first level of fine data
            all_index = np.where(myDelta > 1.0)[0].tolist()

            temp1 = np.logical_and(myDelta > 1.0, myDelta < 2.0)
            temp2 = myDelta > 2.0
            midT1 = (self._tstart[temp1] + self._tstop[temp1]) / 2.0

            for indx in np.where(temp2)[0]:
                for x in midT1:
                    if self._tstart[indx] < x < self._tstop[indx]:

                        try:

                            all_index.remove(indx)

                        except:
                            pass

            all_index = np.array(all_index)

        # Now dump the indices we do not need
        self._tstart = self._tstart[all_index]
        self._tstop = self._tstop[all_index]
        self._qauts = self._qauts[all_index]
        self._sc_pos = self._sc_pos[all_index]
        self._rates = self._rates[all_index, :, :]

        # Now we need to sort because GBM may not have done this!

        sort_mask = np.argsort(self._tstart)
        self._tstart = self._tstart[sort_mask]
        self._tstop = self._tstop[sort_mask]
        self._qauts = self._qauts[sort_mask]
        self._sc_pos = self._sc_pos[sort_mask]
        self._rates = self._rates[sort_mask, :, :]

