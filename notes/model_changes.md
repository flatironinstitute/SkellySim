# Numeric differences between this and Gokberk's version

* Slenderness parameter 'epsilon' (r/L)
  * Value was fixed in original simulations (usually 1E-3)
  * Now it varies as it should (r/L)
* Fix issue with stresslet improperly scaling inversely with viscosity
  Wasn't a problem if eta=1, but otherwise both SkellySim and its
  predecessor were wrong. Fortunately we always used eta=1.
* c_1 for fibers (1 - Xs Xs) cofactor
  Was 2 / 8 / pi / eta
  Now zero by default (fiber.use_local_SBT=False to re-enable)
