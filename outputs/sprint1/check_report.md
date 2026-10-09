# Sprint 1: automatic check report

- SKU rows: 17,125 · colourways: 10,555 · with image file: 9,458
- Colourways needing review: high 5, medium 448

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
| IMG_FILE_MISSING | info | 1097 | 10.39 |
| SALES_MISSING | info | 370 | 3.51 |
| IMG_GENERIC | medium | 249 | 2.36 |
| SKU_ATTR_CONFLICT | medium | 165 | 1.56 |
| IMG_SHARED | medium | 100 | 0.95 |
| CLR_NOT_IN_DESC | medium | 25 | 0.24 |
| SALES_ZERO_QTY | low | 9 | 0.09 |
| IMG_CAT_MISMATCH | medium | 3 | 0.03 |
| TYPE_CONFLICT_DESC | high | 2 | 0.02 |
| IMG_PATH_INVALID | high | 1 | 0.01 |
| IMG_COLOUR_MISMATCH | high | 1 | 0.01 |
| TAB_KEY_INCONSISTENT | high | 1 | 0.01 |
| CLR_CONFLICT_DESC | high | 1 | 0.01 |
| IMG_UNREADABLE | high | 1 | 0.01 |

## Priority by category

| CAT_DES_EN | high | info | low | medium | none |
|---|---|---|---|---|---|
| Apparel | 1 | 314 | 7 | 217 | 1720 |
| Footwear | 0 | 12 | 0 | 16 | 407 |
| Hand Bag | 1 | 156 | 0 | 45 | 1769 |
| Jewellery | 2 | 454 | 0 | 135 | 4048 |
| Wallet | 1 | 39 | 0 | 35 | 1176 |

## Missing values

Full table: `missing_values.csv`. Empty or constant columns (33): INFO_TXT, DIMENSION_CONSUMABLE, END_DATE, AVERAGE_STORE, STATUS_ARTICLE, CAPACITY, SECONDARY_DISPLAY, ITEM_GROUP, FLAG_FACT, PRODUCTION_TYPE, STORE_LOCATION, STORE_CONCEPT, PHASE_COD, CLR_BAS_DES, CLR_TON_DES, PONTAMETALICA_COD, PONTAMETALICA_DES, DISPLAY_CLR, PROD_DAT_CRI, PHASE, DISPLAY_CLR_COD, NIGHT_BAG, ECI_APPAREL, TAR_COD, PROD_SIT, OUTLET_GEN, OUTLET, PERMANENT, TAR_DES, CAT_ACTIVE, ACTIVE_RECORD, PERMANENT_ASIS, WITH_ORDER.

## Examples

**IMG_GENERIC**

| PROD_CLR_EQUIV | PROD_DES_BASE | CLR_DES | GFA_DES_EN | PROG_IMAGE |
|---|---|---|---|---|
| 1619050BM | Earrings MELROSE | Bright Multicolor | Earrings | /182/52/161905_1 |
| 166363_PK | Fan FASHION SUPPLEMENTS Pink | Pink | Other Jewellery | /191/52/166363_1 |
| 166730_AU | Earring GLEAM COLOR | Aubergine | Earrings | /191/52/166730_1 |
| 169622_PK | Earring WHITE FIELDS | Pink | Earrings | /192/52/169622_1 |
| 170562_LK | Earring ORPHIC Light Pink | Light Pink | Earrings | /192/52/170562_1 |

**SKU_ATTR_CONFLICT**

| PROD_CLR_EQUIV | PROD_DES_BASE | CLR_DES | GFA_DES_EN | PROG_IMAGE |
|---|---|---|---|---|
| 200819_CA | Boots LEATHER COWBOY BOOT Camel | Camel | Boots | /232/53/200819_CA_1 |
| 207269_EC | Dress NOTO Ecru | Ecru | Dress | /232/64/207269_EC_1 |
| 208203_EC | T-shirt ESSENTIALS T DISPLAY 1 Ecru | Ecru | T-shirt | /231/64/208203_EC_1 |
| 209638_BN | Dress ESSENTIALS T DISPLAY 1 Brown | Brown | Dress | /232/64/209638_BN_1 |
| 209638_FU | Dress ESSENTIALS T DISPLAY 1 Fuchsia | Fuchsia | Dress | /232/64/209638_FU_1 |

**IMG_SHARED**

| PROD_CLR_EQUIV | PROD_DES_BASE | CLR_DES | GFA_DES_EN | PROG_IMAGE |
|---|---|---|---|---|
| 1930091BM | Phone case FASHION SUPPLEMENTS Bright Multicolor | Bright Multicolor | Mobile Accessories | /232/52/1930091BM_1 |
| 1930092BM | Phone case FASHION SUPPLEMENTS Bright Multicolor | Bright Multicolor | Mobile Accessories | /232/52/1930092BM_1 |
| 208038_BL | Dress ESSENTIALS T DISPLAY 1 Blue | Blue | Dress | /231/64/208038_BL_1 |
| 208038_PU | Dress ESSENTIALS T DISPLAY 1 Purple | Purple | Dress | /232/64/208038_PU_1 |
| 208071_EC | Wallet SAMANTHA Ecru | Ecru | Wallet | /232/62/208071_1 |

**CLR_NOT_IN_DESC**

| PROD_CLR_EQUIV | PROD_DES_BASE | CLR_DES | GFA_DES_EN | PROG_IMAGE |
|---|---|---|---|---|
| 1619050BM | Earrings MELROSE | Bright Multicolor | Earrings | /182/52/161905_1 |
| 166730_AU | Earring GLEAM COLOR | Aubergine | Earrings | /191/52/166730_1 |
| 169622_PK | Earring WHITE FIELDS | Pink | Earrings | /192/52/169622_1 |
| 172276_PP | Earring FANCY PEARLS | Papaya | Earrings | /242/52/172276_PP_1 |
| 176549_BK | Earring WILD COLOR | Black | Earrings | /201/52/176549_1 |

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