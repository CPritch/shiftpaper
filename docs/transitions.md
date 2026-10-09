# Transitions

When the wallpaper changes, from `shiftpaper set` or a slideshow, it plays one of the following transitions listed below based on your config.

```
shiftpaper transition portal      # use portal
shiftpaper transition --secs 2    # take 2 seconds instead of 3
```

Each clip below goes from one wallpaper to the other and back, using the default 3 seconds.

| Transition | |
|---|---|
| **`sweep-in`** (the default)<br><br>A wave sweeps into the scene. The new wallpaper's nearest things appear first, and it fills in towards the distance. | <img src="media/sweep-in.avif" width="400" alt="sweep-in"> |
| **`sweep-out`**<br><br>The old wallpaper's background gives way first, and its nearest things go last. | <img src="media/sweep-out.avif" width="400" alt="sweep-out"> |
| **`morph`**<br><br>Everything changes together, the shape of the scene a little ahead of its colours. | <img src="media/morph.avif" width="400" alt="morph"> |
| **`dissolve`**<br><br>The new wallpaper grows like spheres from random places in the scene, more and more of them as it goes. Further things tend to go first. | <img src="media/dissolve.avif" width="400" alt="dissolve"> |
| **`portal`**<br><br>The new wallpaper grows out from your cursor like a sphere. It spreads over whatever is under the cursor first, then reaches things nearer and further away. It follows the cursor as it moves, so you can paint it in. | <img src="media/portal.avif" width="400" alt="portal"> |
| **`tide-in`**<br><br>The new wallpaper rises like water, filling the lowest places first. | <img src="media/tide-in.avif" width="400" alt="tide-in"> |
| **`tide-out`**<br><br>The old wallpaper drains away like water, uncovering the highest places first. | <img src="media/tide-out.avif" width="400" alt="tide-out"> |

<sub>Photos by [Charles Black](https://commons.wikimedia.org/wiki/File:Rotting_leaves_on_a_forest_path_(Unsplash).jpg) and [Zeny Rosalina](https://commons.wikimedia.org/wiki/File:Smooth_Round_Rocks_Ocean_(Unsplash).jpg), CC0 via Wikimedia Commons.</sub>
