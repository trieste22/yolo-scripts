# Models
### PNRE
##### See (on Trieste’s local rn) /rats/predicts for fullmin, easierjunk, and short predicting on each other -- proof that shorter training file = higher precision. Note that iterations took ~30s-5 minutes
* **“Fullmin”**
  * pnre-junky:/media/PNRE/noisy/fullmin 
  * PNRE 22 species, 8,300 1 min scapes with lots of junk noise (inc humans & bugs)
  * ~ 1.5 days training, over 2019/04/18 and 2019/04/23
  * Loss ~ 7.15
* **“Easierjunk”**
  * pnre-darknet:/media/PNRE/noisy/easierjunk
  * PNRE 22 species, 1 min scapes with easier junk (no humans or bugs) & quieter? backgrounds than “fullmin”
  * ~ 1 day training on 2019/04/23-24
  * Loss ~5.7
* **“Pnre-short”**
  * pnre-short:/media/PNRE/pnre-short
  * PNRE 22 species, 15s scapes with easy junk (on par with “easierjunk”)
  * ~1 day training, 2019/04/23-24
  * Loss ~1.08
  * Solid results, even when tested on “fullmin” and “easierjunk”
* **“pnre-real”**
  * pnre-short:/media/PNRE/pnre-real

### BAMBOO RATS
* **“rats-15s”**
  * pnre-short:/media/rats/rats-15s
  * 10,000 15s scapes with both farinosas and no farinoras, junk added (one big csv file: rats-15s_scapes.csv)
  * Only able to get max 76% anchor accuracy (as compared to (85-90+% for PNRE models)
  * Overnight training (~12 hours?), 2019/05/06-07
  * Learning went down to 0 after a few iterations, and average loss increased by of training
  * Did not attempt any predictions
* **“sparse-rats-5s”**
  * pnre-short:/media/rats/sparse-rats-5s
  * 21870 5s scapes with no farinosas & less, easier junk than “rats-15s” (no bugs)
  * ~18 hrs training, 2019/05/09-10 -> didn’t get anywhere!
  * -nans for ⅔ regions (as compared to for ⅓ regions for healthy PNRE training)
  * Loss started at 2068, only decreased to 2038 after a day
  * Learning rate dropped from .001 to .000000 after first iteration, then stayed there
  * Cancelled to try easier scapes - did not attempt any predictions.
* **“easy-rats-5s”**
  * pnre-short:/media/rats/easy-rats-5s
  * 25,000 5s scapes with no farinosas, high SNR and no junk (one csv per scape)
  * Ran the following to correct for scape gen error:
    * ‘for F in easy-rats-5s_scapes/*.csv; do sed -i "s/,End/Index,End/" $F ;done’
  * 0/10 failure on gen_anchors, average ~89%, best 91.3
  * Note: they look remarkably similar to the anchors generated  for sparse-rats-5s
  * Started training @ 12:10 am 2019/05/10
    * no iterations done by 1:52pm..
  * Stopped training ~12pm 2019/05/13
    * no signs of progress - only progressed through 25 iterations, learning rate still 0.00000, loss stayed about constant
    * Need to find a bug somewhere! Decided to try some rat scapes using an old PNRE setup.
 * **"easy-rats-5s-same"**
   * pnre-short:/media/rats/easy-rats-5s-same
   * one 5-second, single-bark rat file repeated 10,000 times (high SNR & no junk noise - same scape-gen as "easy-rats-5s")
   * anchor_gen failed - only nan outputs. Ran faster than usual. Tried ~15 times.
   * Couldn't go further.. - should look into gen_anchor and figure out what's going on that it would fail like this, and why on a more normal training it's still ~1-2/10 likely to fail..
 * **"easy-rats-5s, squished"
   * pnre-short:/media/rats/easy-rats-5s/squished
   * easy-rats-5s files unchanged, but with YOLO resizing down to weight = 832, height = 256 (in cfg and anchor_gen)
   * Training fail, like other rat failures - learning rate 0, loss actually increasing slightly over time, slow iterations.
 * **"easy-rats-5s, subdiv8"
   * pnre-short:media/rats/easy-rats-5s, subdiv8
   * easy-rats-5s files unchanged, but with subdivisions = 8 (instead of 16 that it's been consistently for other models)
   * Training fail, like other rat failures - learning rate 0, loss increased a lot over time (after initial few iters of ~constant loss). Smaller subdivision resulted in iterations going faster as compared to other easy-rat models, but still quite slow compared to successful PNRE models' iteration times.

