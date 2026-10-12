# Phase 1: automatic check report

- SKU rows: 17,125 · colourways: 10,555 · with image file: 9,491
- Colourways needing review: high 29, medium 567

## Duplicates and keys

| check | count |
|---|---|
| product: fully duplicated rows | 0 |
| product: duplicated PROD_SPK | 0 |
| product: duplicated PROD_COD (SKU) | 0 |
| product: duplicated BAR_COD | 0 |
| product: duplicated PROD_COD_EQUIV (re-coded items, expected) | 28 |
| sales: duplicated keys | 0 |
| sales: keys not in product table | 0 |
| product: colourways without sales | 370 |

## Issues per colourway

| issue | severity | n_items | pct_items |
|---|---|---|---|
| IMG_FILE_MISSING | info | 1064 | 10.08 |
| SALES_MISSING | info | 370 | 3.51 |
| IMG_NOT_PACKSHOT | low | 272 | 2.58 |
| IMG_GENERIC | medium | 216 | 2.05 |
| VIS_TYPE_DOUBT | low | 189 | 1.79 |
| PRICE_COST_MISSING | info | 175 | 1.66 |
| SKU_ATTR_CONFLICT | medium | 165 | 1.56 |
| VIS_COLOUR_MISMATCH | medium | 135 | 1.28 |
| IMG_SHARED | medium | 96 | 0.91 |
| IMG_NEAR_DUPLICATE | medium | 54 | 0.51 |
| VIS_TARGET_NOT_FOUND | low | 46 | 0.44 |
| IMG_SMALL | info | 45 | 0.43 |
| IMG_PATH_REPAIRED | medium | 33 | 0.31 |
| CLR_NOT_IN_DESC | low | 25 | 0.24 |
| VIS_TYPE_MISMATCH | high | 24 | 0.23 |
| REALISED_PRICE_ODD | info | 21 | 0.2 |
| IMG_ODD_SHAPE | info | 15 | 0.14 |
| PRICE_BELOW_COST | info | 10 | 0.09 |
| SALES_ZERO_QTY | low | 9 | 0.09 |
| IMG_CAT_MISMATCH | medium | 3 | 0.03 |
| TYPE_CONFLICT_DESC | high | 2 | 0.02 |
| IMG_PATH_INVALID | high | 1 | 0.01 |
| IMG_COLOUR_MISMATCH | high | 1 | 0.01 |
| TAB_KEY_INCONSISTENT | high | 1 | 0.01 |
| CLR_CONFLICT_DESC | high | 1 | 0.01 |
| SALES_AMT_WITHOUT_QTY | info | 1 | 0.01 |
| IMG_UNREADABLE | high | 1 | 0.01 |

## Priority by category

| CAT_DES_EN | high | info | low | medium | none |
|---|---|---|---|---|---|
| Apparel | 7 | 318 | 25 | 234 | 1675 |
| Footwear | 2 | 15 | 1 | 18 | 399 |
| Hand Bag | 1 | 164 | 44 | 55 | 1707 |
| Jewellery | 16 | 455 | 234 | 211 | 3723 |
| Wallet | 3 | 42 | 58 | 49 | 1099 |

## Missing values

Full table: `missing_values.csv`. Empty or constant columns (33): INFO_TXT, DIMENSION_CONSUMABLE, END_DATE, AVERAGE_STORE, STATUS_ARTICLE, CAPACITY, SECONDARY_DISPLAY, ITEM_GROUP, FLAG_FACT, PRODUCTION_TYPE, STORE_LOCATION, STORE_CONCEPT, PHASE_COD, CLR_BAS_DES, CLR_TON_DES, PONTAMETALICA_COD, PONTAMETALICA_DES, DISPLAY_CLR, PROD_DAT_CRI, PHASE, DISPLAY_CLR_COD, NIGHT_BAG, ECI_APPAREL, TAR_COD, PROD_SIT, OUTLET_GEN, OUTLET, PERMANENT, TAR_DES, CAT_ACTIVE, ACTIVE_RECORD, PERMANENT_ASIS, WITH_ORDER.

## Sales profile

Per category (`sales_profile.csv`). Outliers: units sold more than 3 IQR below the first or above the third quartile, on the log scale (mostly items that sold very little). `realised_vs_base_median`: realised price (amount / units) over the base price; amounts mix currencies.

| CAT_DES_EN | colourways | with_sales | zero_qty | qty_median | qty_p95 | qty_max | low_outliers | high_outliers | amount_total | realised_vs_base_median |
|---|---|---|---|---|---|---|---|---|---|---|
| Apparel | 2259 | 2082 | 8 | 1155 | 7533 | 64097 | 0 | 0 | 118028206 | 0.824 |
| Footwear | 435 | 425 | 0 | 1760 | 10818 | 19794 | 2 | 0 | 32958075 | 0.817 |
| Hand Bag | 1971 | 1925 | 0 | 4049 | 9069 | 16378 | 43 | 0 | 190835403 | 0.889 |
| Jewellery | 4639 | 4533 | 1 | 2788 | 7265 | 112982 | 117 | 0 | 98839510 | 0.849 |
| Wallet | 1251 | 1220 | 0 | 2720 | 5964 | 12544 | 17 | 0 | 43318038 | 0.881 |
| All | 10555 | 10185 | 9 | 2624 | 8028 | 112982 | 122 | 0 | 483979233 | 0.853 |

## Image sizes

| side | min | 1% | 50% | 99% | max |
|---|---|---|---|---|---|
| img_width | 58 | 479 | 640 | 640 | 4032 |
| img_height | 42 | 640 | 640 | 640 | 3363 |

Colour modes: RGB 9,490. Most photos are 640 x 640; `IMG_SMALL` and `IMG_ODD_SHAPE` are notes only (CLIP pads the crop to a square).

## Examples

**IMG_GENERIC**

| PROD_CLR_EQUIV | PROD_DES_BASE | CLR_DES | GFA_DES_EN | PROG_IMAGE |
|---|---|---|---|---|
| 1619050BM | Earrings MELROSE | Bright Multicolor | Earrings | /182/52/161905_1 |
| 166363_PK | Fan FASHION SUPPLEMENTS Pink | Pink | Other Jewellery | /191/52/166363_1 |
| 166730_AU | Earring GLEAM COLOR | Aubergine | Earrings | /191/52/166730_1 |
| 169622_PK | Earring WHITE FIELDS | Pink | Earrings | /192/52/169622_1 |
| 176549_BK | Earring WILD COLOR | Black | Earrings | /201/52/176549_1 |

**SKU_ATTR_CONFLICT**

| PROD_CLR_EQUIV | PROD_DES_BASE | CLR_DES | GFA_DES_EN | PROG_IMAGE |
|---|---|---|---|---|
| 200819_CA | Boots LEATHER COWBOY BOOT Camel | Camel | Boots | /232/53/200819_CA_1 |
| 207269_EC | Dress NOTO Ecru | Ecru | Dress | /232/64/207269_EC_1 |
| 208203_EC | T-shirt ESSENTIALS T DISPLAY 1 Ecru | Ecru | T-shirt | /231/64/208203_EC_1 |
| 209638_BN | Dress ESSENTIALS T DISPLAY 1 Brown | Brown | Dress | /232/64/209638_BN_1 |
| 209638_FU | Dress ESSENTIALS T DISPLAY 1 Fuchsia | Fuchsia | Dress | /232/64/209638_FU_1 |

**VIS_COLOUR_MISMATCH**

| PROD_CLR_EQUIV | PROD_DES_BASE | CLR_DES | GFA_DES_EN | PROG_IMAGE |
|---|---|---|---|---|
| 158685_MT | Earring SUNSET JUNCTION Mint | Mint | Earrings | /241/52/158685_MT_1 |
| 163517_BK | Earring ECLIPSE Black | Black | Earrings | /232/52/163517_BK_1 |
| 166730_AU | Earring GLEAM COLOR | Aubergine | Earrings | /191/52/166730_1 |
| 166730_BR | Earring FLORABUNDANCE Brick Red | Brick Red | Earrings | /232/52/166730_BR_1 |
| 166730_FG | Earring FLEURSDUMER Forest Green | Forest Green | Earrings | /241/52/166730_FG_1 |

**IMG_SHARED**

| PROD_CLR_EQUIV | PROD_DES_BASE | CLR_DES | GFA_DES_EN | PROG_IMAGE |
|---|---|---|---|---|
| 1930091BM | Phone case FASHION SUPPLEMENTS Bright Multicolor | Bright Multicolor | Mobile Accessories | /232/52/1930091BM_1 |
| 1930092BM | Phone case FASHION SUPPLEMENTS Bright Multicolor | Bright Multicolor | Mobile Accessories | /232/52/1930092BM_1 |
| 208038_BL | Dress ESSENTIALS T DISPLAY 1 Blue | Blue | Dress | /231/64/208038_BL_1 |
| 208038_PU | Dress ESSENTIALS T DISPLAY 1 Purple | Purple | Dress | /232/64/208038_PU_1 |
| 208071_EC | Wallet SAMANTHA Ecru | Ecru | Wallet | /232/62/208071_1 |

**IMG_NEAR_DUPLICATE**

| PROD_CLR_EQUIV | PROD_DES_BASE | CLR_DES | GFA_DES_EN | PROG_IMAGE |
|---|---|---|---|---|
| 195269_LK | Earring TREND Light Pink | Light Pink | Earrings | /221/52/195269_LK_1 |
| 195269_WT | Earring TREND White | White | Earrings | /222/52/195269_WT_1 |
| 206914_GD | Necklace GLDN DEL Gold | Gold | Necklaces | /231/52/206914_GD_1 |
| 206914_SV | Necklace SILVER D Silver | Silver | Necklaces | /231/52/206914_SV_1 |
| 208038_BL | Dress ESSENTIALS T DISPLAY 1 Blue | Blue | Dress | /231/64/208038_BL_1 |

**IMG_PATH_REPAIRED**

| PROD_CLR_EQUIV | PROD_DES_BASE | CLR_DES | GFA_DES_EN | PROG_IMAGE |
|---|---|---|---|---|
| 170562_LK | Earring ORPHIC Light Pink | Light Pink | Earrings | /192/52/170562_1 |
| 193076_PU | Earring FLORABUNDANCE Purple | Purple | Earrings | /221/52/193076_1 |
| 193166_BR | Earring mulstone Brick Red | Brick Red | Earrings | /221/52/193166_1 |
| 1999351HM | Earring mulstone Halftone Multicolor | Halftone Multicolor | Earrings | /222/52/199935_1 |
| 209656_BG | Trainer Shoes RUNNING LIGHT BEIGE Beige | Beige | Trainers | /232/53/209656_1 |

**VIS_TYPE_MISMATCH**

| PROD_CLR_EQUIV | PROD_DES_BASE | CLR_DES | GFA_DES_EN | PROG_IMAGE |
|---|---|---|---|---|
| 196653_MA | Earring TREND | Magenta | Earrings | /221/52/196653_1 |
| 207972_PK | Ballerinas ELASTIC BALLERINA Pink | Pink | Ballerinas | /232/53/207972_PK_1y |
| 209690_BM | Mobile Holder FASHION SUPPLEMENTS Bright Multicolor | Bright Multicolor | Mobile Accessories | /232/52/209690_BM_1 |
| 209709_BM | Mobile Holder FASHION SUPPLEMENTS Bright Multicolor | Bright Multicolor | Mobile Accessories | /232/52/209709_1 |
| 210450_GD | Flat Sandals GOLD TUBULAR SANDAL Gold | Gold | Flat Sandals | /232/53/210450_GD_1 |

**IMG_CAT_MISMATCH**

| PROD_CLR_EQUIV | PROD_DES_BASE | CLR_DES | GFA_DES_EN | PROG_IMAGE |
|---|---|---|---|---|
| 208619_CA | Phone Holder GRACE Camel | Camel | Phone Holder | /232/54/208619_1 |
| 216532_FU | Crossbody Bag EMAN Fuchsia | Fuchsia | Cross | /241/55/216532_FU_1 |
| 217193_EC | Crossbody Bag EMAN Ecru | Ecru | Cross | /241/55/217193_EC_1 |

**TYPE_CONFLICT_DESC**

| PROD_CLR_EQUIV | PROD_DES_BASE | CLR_DES | GFA_DES_EN | PROG_IMAGE |
|---|---|---|---|---|
| 220697_BK | Sac WOODY1 Black | Black | Tote | /241/54/220697_BK_1 |
| 224036_GY | Wallet CORDY Grey | Grey | Others | /242/62/224036_GY_1 |

**IMG_PATH_INVALID**

| PROD_CLR_EQUIV | PROD_DES_BASE | CLR_DES | GFA_DES_EN | PROG_IMAGE |
|---|---|---|---|---|
| 140428_BN | Necklace SILVER PROVISION Brown | Brown | Necklaces | /2015-171/52/140428CT__1 |

**IMG_COLOUR_MISMATCH**

| PROD_CLR_EQUIV | PROD_DES_BASE | CLR_DES | GFA_DES_EN | PROG_IMAGE |
|---|---|---|---|---|
| 140428_BN | Necklace SILVER PROVISION Brown | Brown | Necklaces | /2015-171/52/140428CT__1 |

**TAB_KEY_INCONSISTENT**

| PROD_CLR_EQUIV | PROD_DES_BASE | CLR_DES | GFA_DES_EN | PROG_IMAGE |
|---|---|---|---|---|
| 212653_WT | T-shirt T1 ESSENTIALS White | Ecru | T-shirt | /212/64/192346_EC_1 |

**CLR_CONFLICT_DESC**

| PROD_CLR_EQUIV | PROD_DES_BASE | CLR_DES | GFA_DES_EN | PROG_IMAGE |
|---|---|---|---|---|
| 212653_WT | T-shirt T1 ESSENTIALS White | Ecru | T-shirt | /212/64/192346_EC_1 |

**IMG_UNREADABLE**

| PROD_CLR_EQUIV | PROD_DES_BASE | CLR_DES | GFA_DES_EN | PROG_IMAGE |
|---|---|---|---|---|
| 216726_DM | Earring Gipsy Garden Dark Multicolor | Dark Multicolor | Earrings | /241/52/216726_DM_1 |

## What these checks cannot see

They compare codes and text. They cannot see what an image actually shows (e.g. a necklace row with an earring photo). That needs human review (`batch` command) or a vision model.