# from lightkurve import search_targetpixelfile
import lightkurve as lk
import matplotlib.pyplot as plt 
import pandas as pd 

search_result = lk.search_lightcurve('HD 96715', author='SPOC')
print(search_result)
lc_collection = search_result.download_all()

lc_stitched = lc_collection.stitch()
lc_stitched.plot(linewidth=0, marker='.')
print(lc_stitched.time)
print(lc_stitched.flux)
plt.savefig('lccollectiontest.png')
plt.clf()

periodogram = lc_stitched.to_periodogram()
print('frequencies', periodogram.frequency)
print('power', periodogram.power)
df = pd.DataFrame(columns=['Frequency', 'Power'])
df['Frequency'] = periodogram.frequency.value
df['Power'] = periodogram.power.value
df.to_hdf('lightkurvefreqpow.h5', key='df', mode='w')
ax = periodogram.plot()
ax.set_yscale('log')
ax.set_xscale('log')
plt.savefig('periodogramtest.png')

lc_test = lc_collection[0]
lc_test.normalize().plot(linewidth=0, marker='.')
plt.savefig('lc_test.png')
plt.clf()
pd_test = lc_test.to_periodogram()
ax_test = pd_test.plot()
ax_test.set_yscale('log')
ax_test.set_xscale('log')   
plt.savefig('pd_test.png')

# tutorial code
# pixelfile = search_targetpixelfile("KIC 8462852", quarter=16).download()
# pixelfile.plot(frame=1)
# lc = pixelfile.to_lightcurve(aperture_mask='all')
# print(lc.time)
# print(lc.flux)
# lc.plot()
# plt.savefig('lightcurvtest.png')
# pd = lc.to_periodogram()
# plt.clf()
# pd.plot()
# plt.savefig('periodogramtest.png')