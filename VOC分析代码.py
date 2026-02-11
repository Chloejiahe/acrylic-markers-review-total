import pandas as pd
import numpy as np
import os, re
import streamlit as st
from textblob import TextBlob
import nltk
from nltk.tokenize import sent_tokenize
import plotly.graph_objects as go 
from plotly.subplots import make_subplots
from wordcloud import WordCloud, STOPWORDS
import matplotlib.pyplot as plt

# --- 必须添加的内容：资源初始化 ---
def init_nltk_resources():
    resources = ['punkt', 'punkt_tab']
    for res in resources:
        try:
            # 检查资源是否存在
            nltk.data.find(f'tokenizers/{res}')
        except LookupError:
            # 不存在则下载
            nltk.download(res)

init_nltk_resources()

# --- 1. 核心词库配置 (Feature Keywords) ---
FEATURE_DIC = {
        '颜色种类': {
            '正面-色彩丰富': ['many colors', 'lot of colors', 'plenty of colors', 'good range', 'great variety', 'great selection', 'every color', 'all the colors', 'so many options'],
            '负面-色彩单调/反馈': ['limited range', 'not enough colors', 'wish for more', 'missing colors', 'disappointed with selection', 'needs more colors'],
            '正面-套装/数量选择满意': ['love the large set', 'great number of colors', 'perfect amount of colors', 'huge set of 72', 'full set is amazing', 'good assortment'],
            '负面-套装/数量选择不满意': ['wish for a smaller set', 'too many colors', 'no smaller option', 'forced to buy the large set', 'have to buy the whole set'],
            '正面-色系规划满意': ['great color selection', 'perfect pastel set', 'good range of skin tones', 'well-curated palette', 'love the color story', 'beautiful assortment of colors', 'has every color I need'],
            '负面-色系规划不满': ['missing key colors', 'no true red', 'needs more grays', 'too many similar colors', 'palette is not useful', 'wish it had more pastels', 'poor color selection', 'needs more skin tones'],
            '负面-色系失衡':['too many similar colors', 'missing greens', 'needs more warm tones', 'repeated shades', 'doubles of colors', 'lack of blue range'],
            '中性-提及色彩丰富度': ['color range', 'color variety', 'color selection', 'number of colors', 'range of shades','selection of hues', 'color assortment', 'color palette', 'spectrum of colors', 'array of colors', 'how many colors'],

            },
        '色彩一致性': {
            '正面-颜色准确': ['as shown in the photo', 'true to color', 'match the cap', 'accurate color', 'color accuracy', 'exact color', 'matches perfectly', 'consistent color', 'consistency'],
            '负面-颜色偏差': ['color doesn\'t match', 'shade doesn\'t match', 'misleading cap color', 'wrong cap color', 'color is off', 'darker than cap', 'lighter than cap', 'wrong color', 'color different', 'shade is different','red looks pink', 'green is lighter', 'orange looks like peach', 'gold has a greenish tone', 'darker than expected'],
            '正面-设计-颜色准确 (VS 笔帽)': ['true to cap','true to color', 'match the cap', 'matches the cap perfectly', 'cap is a perfect match', 'cap is accurate'],
            '负面-设计-颜色误导 (VS 笔帽)': ['cap is a hint', 'cap is misleading', 'not even close to the cap', 'cap color is off', 'different from the lid', 'misleading cap', 'cap is wrong', 'cap is a lie', "color doesn't match the barrel", 'the cap color is way off', 'nothing like the cap'],
            '正面-营销-颜色准确(VS 网图)': ['matches the description', 'exactly as advertised', 'what you see is what you get', 'matches the online photo', 'true to the swatch', 'photo is accurate'],
            '负面-营销-图片误导 (VS 网图)': ['looks different from the online swatch', 'not the color in the picture', 'misrepresented color', 'photo is misleading', 'swatch card is inaccurate'],
            '正面-生产-品控(VS 其他笔)': ['consistent color', 'consistency', 'no variation between pens', 'reliable color', 'batch is consistent'],
            '负面-生产-品控偏差(VS 其他笔)': ['inconsistent batch', 'color varies from pen to pen', 'my new pen is a different shade', 'no quality control', 'batch variation'],
            '中性-提及笔帽颜色': ['cap color', 'barrel color', 'match the cap', 'color of the cap', 'color on the barrel','cap match', 'indicator on the cap', 'swatch on the barrel', 'color indicated on the pen'],
            '中性-提及网图/色卡': ['swatch card', 'color swatch', 'online swatch', 'swatching', 'swatch test', 'product photo', 'online photo', 'listing photo', 'website image', 'advertised picture', 'photo in the listing'],

     },
        '色彩饱和度与混合': {
            '正面-鲜艳/饱和': ['bright colors', 'nice and bright', 'beautifully bright', 'richly saturated', 'perfectly saturated', 'deeply saturated', 'nice saturation', 'vibrant colors', 'rich colors', 'colors pop'],
            '负面-太鲜艳/刺眼': ['garish colors', 'colors are too loud', 'too neon', 'too bright', 'too fluorescent', 'overly bright'],
            '负面-暗淡/褪色': ['dull', 'faded', 'pale', 'washed out', 'not bright', 'too pale', 'lackluster', 'colors are too dull', 'muddy colors', 'colors look dirty', 'desaturated','doesn\'t show up well',],
            '正面-遮盖力强': ['great coverage', 'completely opaque', 'one coat covers all', 'thick paint feel', 'works on rocks', 'shows up on black', 'vibrant on dark surfaces', 'solid color', 'creamy texture'],
            '负面-遮盖力差': ['too transparent', 'very sheer', 'watery', 'streaky', 'takes many coats', 'see through', 'thin ink', 'runny', 'faint colors'],
            '中性-提及饱和度': ['saturation', 'vibrancy', 'color intensity', 'richness of color', 'color depth', 'pigment load', 'high saturation', 'low saturation', 'deep saturation'],
            '正面-叠色顺滑': ['layers well', 'easy to layer', 'good layering', 'smooth layering', 'layers perfectly', 'buildable color', 'smooth finish', 'not streaky', 'even layers', 'consistent texture', 'creamy application', 'doesn\'t disturb bottom layer', 'sits on top nicely', 'doesn\'t lift previous paint', 'no bleeding between layers'],
            '负面-叠色困难': ['takes forever to dry before layering', 'smears the bottom color', 'lifts if not 100% dry', 'lifts the layer underneath', 'scratches the paint off', 'rubs off previous layer', 'strips the paint', 'disturbs dried paint', 'tears up the bottom color', 'too transparent', 'very sheer', 'streaky when layering', 'can see the color underneath', 'takes too many coats to cover', 'patchy coverage' ],

        },        
        '色系评价': {
            '正面-喜欢标准/基础色系': ['good standard colors', 'love the basic set', 'has all the primary colors', 'classic colors',  'great essential colors', 'perfect starter palette', 'all the fundamental colors',
                          'love the traditional colors', 'a solid basic palette', 'includes primary and secondary colors','just the basics I needed', 'don\'t need anything fancy'],
            '正面-喜欢鲜艳/饱和色系': ['love the vibrant colors', 'love the rich colors', 'love the bold colors', 'love how vivid the colors are','highly saturated', 'nicely saturated', 'colors are saturated', 'great saturation', 'amazing saturation', 'deeply saturated',
                             'colors pop', 'really pop', 'makes the colors pop', 'colors really stand out', 'colors jump off the page','not dull at all', 'anything but muted', 'so full of life'],
            '正面-喜欢粉彩色/柔和系': ['love the pastel colors', 'beautiful pastels', 'adore the soft colors', 'perfect muted tones','calming color palette', 'soothing shades', 'gentle on the eyes', 'doesn\'t hurt my eyes',
                          'subtle and elegant', 'love the macaron colors', 'the muted palette is gorgeous','unlike neon', 'unlike fluorescent', 'not overwhelming'],
            '正面-喜欢复古/怀旧色系': ['love the vintage colors', 'perfect retro palette', 'nostalgic color scheme', 'love the old school colors','adore the vintage feel', 'love the retro vibe', 'perfect nostalgic feel',
                          'great aged palette', 'beautiful antique colors', 'love the heritage colors','70s color palette', 'mid-century modern colors','love the mustard yellow', 'love the avocado green', 'love the burnt orange'],
            '正面-喜欢莫兰迪色系': ['love the morandi colors', 'adore the morandi palette', 'perfect morandi palette', 'beautifully dusty colors', 'love the grayish tones', 'muted and elegant', 'sophisticated colors',
                        'calming color palette', 'soothing to the eye', 'understated and beautiful','looks so high-end', 'elegant color scheme', 'love the muted aesthetic'],
            '正面-喜欢中性/肤色系': ['love the skin tones', 'great range of skin tones', 'perfect for portraits', 'excellent for portraiture','realistic skin tones', 'beautiful flesh tones', 'wide variety of skin colors',
                         'perfect neutral palette', 'love the neutral colors', 'great selection of neutrals', 'beautiful earth tones','adore the earthy palette', 'good selection of beiges', 'love the taupes'],
            '正面-喜欢大地/自然色系': ['love the earth tones', 'adore the earthy palette', 'beautiful natural colors', 'gorgeous nature-inspired colors', 'love the botanical colors', 'perfect botanical palette', 'beautiful forest greens', 'love the sage green',
                          'stunning desert tones', 'love the terracotta shades', 'beautiful clay tones','gorgeous ocean blues', 'love the coastal colors', 'calming mountain palette'],
            '正面-喜欢灰色系': ['love the gray scale', 'great set of cool grays', 'perfect warm grays', 'good neutral grays','excellent grayscale palette', 'adore the range of grays', 'beautiful selection of grays',
                       'love the different shades of gray', 'perfect for monochromatic work', 'great for shadows and shading','the warm grays are beautiful', 'the cool grays are perfect'],
            '正面-喜欢季节/主题色系': ['love the seasonal set', 'perfect seasonal palette', 'beautiful themed set', 'gorgeous forest colors', 'love the ocean tones', 'stunning coastal palette',
                          'perfect autumn palette', 'love the fall colors', 'beautiful autumn shades','vibrant spring colors set', 'love the spring palette', 'beautiful summer tones', 'perfect tropical palette',
                         'love the winter palette', 'gorgeous wintery shades','perfect for christmas', 'great holiday color scheme', 'love the halloween set'],
            '正面-喜欢霓虹/荧光色系': ['love the neon colors', 'beautiful neon colors', 'amazing fluorescent palette', 'super bright neon','the neon really pops', 'vibrant neon shades', 'glows under blacklight', 'perfect for blacklight art',
                          'electric colors are stunning', 'day-glo colors are so bright'],
            '正面-喜欢金属/珠光色系': ['love the metallic colors', 'great metallic effect', 'beautiful metallic sheen', 'shiny metal finish','gorgeous chrome finish', 'looks like real metal', 'love the pearlescent finish', 'beautiful shimmer',
                          'amazing liquid chrome effect', 'very reflective', 'stunning iridescent colors', 'love the lustre','the gold is so rich', 'the silver is brilliant', 'beautiful copper tone'],
            '负面-色系搭配不佳': ["palette is ugly", "colors don't go well together", 'weird color combination', 'unusable colors in set','poorly curated', 'terrible color choices', 'bad color selection', 'the colors clash',
                        'no color harmony', 'strange selection of colors', 'too many similar colors', 'missing key colors', 'not a cohesive palette', 'thoughtless color selection', 'colors are jarring'],
            '中性-提及标准/基础色系': ['standard colors', 'basic set', 'primary colors', 'secondary colors', 'classic colors',
                          'essential colors', 'core colors', 'fundamental palette', 'starter set of colors','introductory palette', 'traditional colors',
                          'basic color wheel'],
            '中性-提及鲜艳/饱和色系': ['vibrant colors', 'bright colors', 'bold colors', 'rich colors', 'vivid colors','color intensity', 'richness of color', 'deep colors','intense colors', 'highly saturated', 'brilliant colors', 'bold palette',
                          'strong colors', 'pop of color', 'jewel tones'],
            '中性-提及粉彩色/柔和系': ['pastel', 'pastels', 'pastel colors', 'pastel shades', 'soft colors', 'subtle shades', 'mild colors', 'macaron colors', 'muted tones',
                       'muted colors', 'gentle colors','delicate colors', 'light colors', 'pale palette', 'low saturation', 'desaturated colors','baby colors',
                       'baby pink', 'baby blue', 'mint green', 'lavender','sorbet colors', 'ice cream colors', 'easter colors'
            ],
            '中性-提及复古/怀旧色系': ['vintage colors', 'retro palette', 'nostalgic colors', 'old school colors', 'sepia tones','aged palette', 'antique colors', 'heritage colors', 'desaturated palette',
                           'mid-century modern colors', 'dusty rose', 'mustard yellow', 'avocado green'],
            '中性-提及莫兰迪色系': [ 'morandi', 'morandi colors', 'morandi palette', 'dusty colors', 'grayish tones', 'muted palette', 'hazy colors', 'understated colors', 'sophisticated palette',
                          'low saturation colors', 'dusty pink', 'sage green', 'pale blue', 'terracotta', 'beige tones'],
            '中性-提及中性/肤色系': ['skin tones', 'flesh tones', 'skin tone palette', 'portrait palette', 'range of skin tones',
                          'neutral palette', 'neutral colors', 'neutrals', 'set of neutrals', 'earth tones'],
            '中性-提及大地/自然色系': ['earth tones', 'earthy palette', 'natural colors', 'nature-inspired colors',  'botanical colors', 'botanical palette', 'forest greens', 'sage green', 'olive green', 'moss green',
                          'desert tones', 'canyon colors',  'clay tones','ocean blues', 'coastal colors', 'seafoam green', 'sky blue','mountain palette', 'stone grays', 'slate gray'],
            '中性-提及灰色系': ['gray scale', 'grayscale', 'grays', 'shades of gray', 'set of grays','cool grays', 'warm grays', 'neutral grays', 'french grey', 'payne\'s grey','light gray', 'dark gray', 'charcoal gray', 'slate gray', 'silver gray', 'monochromatic palette'],

            '中性-提及霓虹/荧光色系': ['neon colors', 'neon palette', 'neon set', 'fluorescent colors', 'fluorescent shades',
                          'highlighter colors', 'day-glo colors', 'electric colors', 'glow under blacklight', 'blacklight reactive', 'uv reactive ink'],
            '中性-提及金属/珠光色系': ['metallic ink', 'metallic colors', 'metallic finish', 'metallic sheen', 'liquid chrome', 'mirror finish','pearlescent effect', 'pearlescent finish', 'mother-of-pearl effect',
                          'shimmering ink', 'shimmer effect', 'glittering effect', 'lustre', 'iridescent','gold ink', 'silver ink', 'bronze ink', 'copper ink'],
            '中性-提及色系搭配': ['color palette', 'color combination', 'color scheme', 'color story', 'color assortment', 'curated palette', 'well-chosen colors', 'colors work together', 'color harmony',
                       'range of colors', 'selection of colors'],
        },
        '笔头表现': {
            '正面-双头设计': ['love the dual tip', 'love two tips', 'convenient dual', 'brush and fine combo', 'best of both worlds'],
            '负面-双头设计': ['useless dual', 'redundant tip', 'unnecessary side', 'wish it was single', 'only use one side'],
            '正面-软头/笔尖好': ['love the brush', 'great brush', 'responsive nib', 'flexible tip', 'smooth application'],
            '负面-笔头磨损分叉': ['tip frays', 'frayed', 'split nib', 'wore out', 'lost its point', 'clogged', 'nib broke'],
            '正面-细节控制好': ['precise fine', 'perfect for details', 'crisp line', 'intricate work', 'sharp chisel', 'sturdy bullet'],
            '负面-细节控制差': ['scratchy', 'too broad', 'too thick', 'dried out', 'lost its edge', 'skips', 'bent the tip'],
            '正面-按压泵吸顺畅': ['easy to prime', 'primed quickly', 'fast to start', 'instant flow', 'smooth pump', 'ready in seconds'],
            '负面-按压泵吸差/漏墨': ['hard to prime', 'impossible to start', 'pumping for 10 minutes', 'stuck valve', 'nib receded', 'ink gushed', 'massive blob', 'messy splattered'],
            '正面-弹性/可替换': ['flexible', 'bouncy', 'nice spring', 'replaceable nibs', 'can replace tips'],
            '负面-硬度/不可替换': ['too stiff', 'too soft', 'mushy', 'no replacement', "can't replace"],
            '中性-提及笔头': ['dual tip', 'brush tip', 'fine liner', 'chisel nib', 'bullet point', 'dot marker', 'line variation', '0.5mm']
        },
        '笔头耐用性': {
            '正面-耐磨损/抗分叉': ["doesn't fray", 'no fraying', 'resists fraying', 'not splitting', 'still intact', 'no signs of wear', 'holds up'],
            '负面-磨损/分叉': ['fray', 'fraying', 'frayed tip', 'split nib', 'splitting', 'wears out quickly', 'wear down fast', 'tip is gone'],
            '正面-保形/硬度佳': ['retains shape', 'holds its point', 'stays sharp', "doesn't get mushy", "doesn't go flat", 'springs back', 'good snap'],
            '负面-形变/软化': ['gets mushy', 'too soft', 'tip softened', 'spongy', 'lost its point', 'point went dull', 'deformed', 'went flat'],
            '正面-坚固抗损': ['durable tip', 'sturdy nib', 'robust', 'heavy duty', 'resilient', 'withstands pressure', "doesn't break"],
            '负面-意外损坏': ['bent tip', 'breaks easily', 'snapped', 'cracked tip', 'chipped', 'damaged', 'tip fell out', 'pushed in', 'receded'],
            '负面-寿命不匹配': ['tip wore out before ink', 'died before the ink', 'ink left but tip is useless', 'nib is gone but still has ink'],
            '正面-寿命长': ['long lasting tip', 'outlasts the ink', 'good longevity', 'lasts a long time']
        },
        '流畅性': {
            '正面-书写流畅': ['writes smoothly', 'buttery smooth', 'glides on surface', 'effortless application', 'flows like a dream', 'like silk', 'no resistance', 'smooth as butter', 
                         'glides across the canvas', 'seamless flow', 'very fluid'],
            '负面-干涩/刮纸/断墨': ['scratchy', 'feels scratchy on paper', 'scratches the paper', 'scratchy nib','writes dry', 'arrived dried out', 'dried up quickly', 'pen is dry', 'ink seems dry',
                         'skips', 'skipping', 'skips constantly', 'ink skips', 'hard start', 'hard starts', 'stops and starts','inconsistent flow', 'uneven ink flow', 'ink flow is not consistent',
                        'stops writing', 'stopped working', 'died after a few uses'],
            '负面-出墨过多/漏墨': [ 'blotchy', 'splotchy', 'leaves ink blots', 'ink blobs','too much ink', 'puts down too much ink', 'gushes ink', 'ink gushes out', 'too wet',
                        'feathers badly', 'bleeds everywhere','leaks', 'leaking ink', 'leaked all over', 'ink leaked everywhere', 'leaky pen', 'arrived leaking'],
            '正面-速干且平整': [ 'dries flat', 'no streaks', 'smooth finish', 'not watery', 'solid line', 'vibrant coverage', 'thick paint feel', 'professional finish', 
                        'uniform texture', 'velvety finish', 'opaque delivery'],
            '负面-漆感差/稀薄/起泡': [ 'watery ink', 'too thin', 'pigment and water separated', 'faint color flow', 'bubbles in paint', 'air bubbles', 'frothy ink', 'see-through lines', 
                        'weak pigment', 'diluted color', 'runny paint'],
        },
        '墨水特性': {
            '正面-干燥快/防涂抹': ['quick dry', 'dry so fast','fast dry','not smear','not bleed','no bleed', 'not smear or bleed','dries quickly', 'dries instantly', 'dries immediately', 'fast-drying ink','no smear', 'no smudge', 'zero smear', 'zero smudge', 'smear proof', 'smudge proof',
                        'smudge resistant', 'smear resistant', 'doesn\'t smear', 'doesn\'t smudge','good for lefties', 'perfect for left-handed', 'lefty friendly','can highlight over it', 'highlight without smearing'],
            '负面-干燥慢/易涂抹': ['smears easily', 'smudges easily', 'smears across the page', 'smudges when touched', 'takes forever to dry', 'long drying time', 'never fully dries', 'still wet after minutes', 'slow to dry',
                        'not for left-handed', 'not for lefties', 'smears for left-handers', 'gets ink on my hand','smears with highlighter', 'smudges when layering', 'ruined my work by smudging'],
            '正面-环保/安全/无味': ['non-toxic', 'AP certified non-toxic', 'certified non-toxic', 'no harmful chemicals', 'acid-free', 'archival quality', 'archival ink', 'photo safe','safe for kids', 'kid-safe', 'child-safe', 'great for children',
                        'no smell', 'no odor', 'odorless', 'scent-free', 'low odor', 'no fumes', 'no harsh smell', 'doesn\'t smell bad', 'xylene-free'],
            '负面-气味难闻': ['bad smell', 'strong smell', 'chemical smell', 'toxic smell', 'horrible odor', 'awful scent','overpowering smell', 'overwhelming fumes', 'nauseating smell', 'smells terrible',
                     'stinks', 'reek', 'stench', 'acrid smell', 'plastic smell','gives me a headache', 'headache inducing', 'smell is too strong', 'lingering smell'],
            '正面-持久/防水': ['truly permanent', 'permanent bond', 'archival quality', 'archival ink', 'museum quality','is waterproof', 'water resistant', 'doesn\'t run with water', 'survives spills', 'water-fast',
                      'fade proof', 'fade resistant', 'lightfast', 'excellent lightfastness', 'uv resistant', 'doesn\'t fade over time'],
            '负面-易褪色/不防水': ['not permanent', 'isn\'t permanent', 'fades quickly', 'fades over time', 'colors have faded', 'not lightfast','not waterproof', 'isn\'t water resistant', 'washes away', 'runs with water', 'smears with water',
                        'ruined by a drop of water', 'ink bleeds when wet'],
            '正面-续航长': ['lasts a long time', 'lasted for months', 'seems to last forever', 'plenty of ink', 'large ink capacity', 'still going strong', 'has a lot of ink', 'haven\'t run out yet',
                    'great longevity', 'long-lasting ink'],
            '负面-消耗快': ['runs out quickly', 'ran out of ink fast', 'dries out too fast', 'died quickly','empty fast', 'used up too fast', 'ran dry very quickly', 'doesn\'t last long','run out of paint',
                     'ink runs out in a day', 'not much ink inside', 'low ink capacity', 'wish it held more ink'],
            '正面-金属效果好': ['great metallic effect', 'nice metallic sheen', 'shiny metal finish', 'strong metallic look', 'looks like real metal', 'beautiful chrome finish', 'very reflective'],
            '负面-金属效果差': ['dull metallic', 'not shiny', 'no metallic effect', 'looks flat', 'weak sheen', 'not reflective'],
            '正面-闪光效果好': ['lots of glitter', 'beautiful shimmer', 'sparkly', 'glitter is vibrant', 'nice pearlescent effect', 'very glittery', 'good sparkle'],
            '负面-闪光效果差': ['not enough glitter', 'no shimmer', 'glitter falls off', 'dull sparkle', 'barely any glitter', 'messy glitter'],
            '正面-荧光/霓虹效果好': ['neon pops', 'very bright neon', 'glows under blacklight', 'super fluorescent', 'vibrant neon', 'glows nicely'],
            '负面-荧光/霓虹效果淡': ['neon is dull', 'not very bright', "doesn't glow", 'not a true neon color', 'disappointing neon'],
            '负面-荧光/霓虹效果过饱和': ['too neon', 'too bright', 'too fluorescent', 'too neon/bright'],
            '正面-变色效果好': ['love the color change', 'chameleon effect is stunning', 'shifts colors beautifully', 'works in the sun', 'heat sensitive works'],
            '负面-变色效果差': ["doesn't change color", 'color shift is weak', 'barely changes', 'no chameleon effect'],
            '正面-夜光效果好': ['glows brightly in the dark', 'long lasting glow', 'charges quickly', 'very luminous'],
            '负面-夜光效果差': ["doesn't glow", 'glow is weak', 'fades too fast', 'barely glows'],
            '正面-香味好闻': ['smells great', 'love the scent', 'nice fragrance', 'fun scents', 'smells like fruit'],
            '负面-香味难闻/太浓': ['smell is too strong', 'bad smell', "doesn't smell like anything", 'chemical smell', 'artificial scent'],
            '中性-提及干燥/涂抹': ['drying time', 'dry time', 'smudge proof', 'smear proof', 'for left-handed', 'for lefties'],
            '中性-提及气味/安全': ['odor', 'smell', 'fumes', 'scent', 'non-toxic', 'acid-free', 'archival quality', 'chemical smell', 'safe for kids'],
            '中性-提及持久/防水': ['waterproof', 'water resistance', 'fade proof', 'lightfastness', 'lightfast rating', 'permanent ink', 'archival ink'],
            '中性-提及续航': ['longevity', 'ink life', 'how long they last', 'runs out quickly', 'runs dry', 'ink capacity'],
            '中性-提及金属效果': ['metallic ink', 'metallic effect', 'metallic sheen', 'chrome finish', 'reflective properties'],
            '中性-提及闪光效果': ['glitter ink', 'shimmer effect', 'sparkle', 'pearlescent effect', 'glitter particles'],
            '中性-提及荧光/霓虹效果': ['neon ink', 'fluorescent colors', 'under blacklight', 'glowing effect'],
            '中性-提及变色效果': ['color changing', 'color shift', 'chameleon effect', 'heat sensitive'],
            '中性-提及夜光效果': ['glow in the dark', 'luminous ink', 'glowing properties'],
            '中性-提及香味': ['scented ink', 'scented markers', 'fruit scent', 'fragrance'],
            '中性-提及可擦除性': ['erasable ink', 'can be erased', 'erases cleanly', 'frixion ink'],
        },
        '笔身与易用性': {
            '正面-材质/做工好': ['durable body', 'sturdy', 'sturdy build', 'well-made', 'solid construction', 'solidly built','quality feel', 'feels premium', 'high quality materials', 'quality build', 'well put together','feels substantial', 'built to last', 'high-grade plastic', 'metal construction', 'feels expensive'],
            '负面-材质/做工差': ['feels cheap', 'flimsy', 'cheap plastic', 'thin plastic', 'brittle plastic', 'feels plasticky', 'poorly made', 'poor construction', 'badly made', 'low quality build', 'fell apart',
                       'cracked easily', 'developed a crack','break', 'broke easily', 'broke when dropped', 'snapped in half', 'easy to break','doesn\'t feel durable', 'not sturdy'],
            '正面-握持舒适': ['comfortable to hold', 'comfortable grip', 'ergonomic', 'ergonomic design', 'ergonomic shape', 'nice to hold', 'feels good in the hand', 'feels great in the hand', 'good grip', 'soft grip',
                      'well-balanced', 'perfect weight', 'nice balance', 'fits my hand perfectly', 'contours to my hand', 'doesn\'t cause fatigue', 'no hand cramps', 'can write for hours', 'can draw for hours', 'reduces hand strain'],
            '负面-握持不适': [ 'uncomfortable to hold', 'uncomfortable grip', 'awkward to hold', 'awkward shape','causes hand fatigue', 'tires my hand quickly', 'gives me hand cramps', 'hand cramps up', 'hurts my hand', 'digs into my hand', 'sharp edges', 'too thick', 'too thin', 'too wide', 'too narrow', 'slippery grip', 'hard to get a good grip', 'poorly balanced', 'too heavy', 'too light', 'weird balance'],
            '正面-笔帽体验好': ['cap posts well', 'posts securely', 'cap posts nicely', 'secure fit', 'cap fits snugly', 'airtight seal', 'seals well', 'tight seal', 'cap clicks shut', 'satisfying click', 'audible click', 'easy to open cap', 'easy to uncap', 'cap stays on', 'doesn\'t dry out', 'durable clip'],
            '负面-笔帽体验差': ['hard to open cap', 'cap is too tight', 'difficult to uncap', 'struggle to open','loose cap', 'cap falls off', "cap doesn't stay on", "doesn't seal properly", 'not airtight',
                       'pen dried out because of the cap', 'dries out quickly','cracked cap', 'cap broke', 'cap broke easily', 'brittle cap','cap won\'t post', 'doesn\'t post securely', 'cap is too loose to post','clip broke off', 'flimsy clip', 'weak clip'],
            '正面-易于使用': ['easy to use', 'simple to use', 'user-friendly', 'intuitive design', 'no learning curve', 'effortless to use', 'easy to handle', 'easy to control', 'good control',],
            '中性-提及材质/做工': ['pen body', 'body material', 'barrel material', 'build quality', 'construction', 'plastic body', 'metal body', 'wooden body', 'resin body', 'surface finish', 'pen finish'],
            '中性-提及握持': ['grip section', 'grip comfort', 'ergonomic grip', 'pen balance', 'pen shape', 'barrel diameter',  'how it feels in the hand'],
            '中性-提及笔帽': ['pen cap', 'pocket clip', 'posting the cap', 'cap seal', 'screw cap', 'snap cap','caps','cap'],
            '中性-提及易用性/便携': ['portability', 'easy to carry', 'travel case', 'pen roll', 'for on the go', 'pocket pen', 'travel friendly'],
        },
        '绘画表现': {
            '正面-线条表现好/可控': ['good control', 'controllable lines', 'great line variation', 'crisp lines', 'consistent lines', 'clean lines', 'no skipping', 'sharp lines', 'great for fine details'],
            '负面-线条表现差/难控': ['hard to control', 'inconsistent line', 'uncontrollable', 'not for details', 'wobbly lines', 'shaky lines', 'broken line'],
            '正面-覆盖力好/不透明': ['opaque', 'good coverage', 'covers well', 'one coat', 'hides underlying color', 'works on dark paper', 'great opacity'],
            '负面-过于透明/覆盖力差': ['not opaque', 'too sheer', "doesn't cover", 'needs multiple coats', 'see through'],
            '正面-涂色均匀': ['even application', 'smooth application', 'no streaks', 'self-leveling', 'consistent color', 'no streaking'],
            '负面-涂色不均': ['streak', 'streaky', 'streaking', 'leaves streaks', 'patchy', 'blotchy'],
            '正面-兼容铅笔': ['goes over pencil cleanly', "doesn't smudge graphite", 'erases pencil underneath', 'covers pencil lines well'],
            '负面-铅笔兼容性差': ['smears pencil lines', 'smudges graphite', 'lifts graphite', 'muddy with pencil', "doesn't cover pencil"],
            '正面-兼容勾线笔': ["doesn't smear fineliner", 'works with micron pens', 'layers over ink', 'copic-proof ink compatible', 'safe over ink'],
            '负面-勾线笔兼容性差': ['smears fineliner ink', 'reactivates ink', 'lifts the ink line', 'bleeding with ink lines', 'makes ink run'],
            '正面-兼容水彩/水粉': ['layers over watercolor', 'works well with gouache', 'can use for watercolor effects', "doesn't lift watercolor"],
            '负面-水彩/水粉兼容性差': ['lifts watercolor', 'muddy with gouache', 'reactivates paint underneath', 'smears watercolor'],
            '正面-兼容彩铅': ['layers well with colored pencils', 'good for marker and pencil', 'blends with pencil crayon', 'works over wax pencil'],
            '负面-彩铅兼容性差': ['waxy buildup with colored pencils', "doesn't layer over pencil crayon", 'reacts weirdly with other markers'],
            '正面-兼容丙烯马克笔': ['layers nicely over Posca', 'can draw on top of Posca', "doesn't lift the acrylic", 'good with acrylic markers', 'adheres well to paint'],
            '负面-不兼容丙烯马克笔': ['smears Posca paint', "doesn't stick to acrylic marker", 'lifts the underlying acrylic', 'scratches off the acrylic surface'],
            '中性-提及线条表现': ['line quality', 'line control', 'line variation', 'stroke consistency', 'stroke', 'fine details', 'detailed work'],
            '中性-提及覆盖力': ['opacity', 'coverage', 'sheer', 'transparency', 'opaque', 'single coat', 'coverage strength'],
            '中性-提及涂色均匀性': ['even application', 'smooth application', 'streaks', 'streaky', 'patchy', 'blotchy', 'self-leveling'],
            '中性-提及可再加墨': ['reactivate', 'reactivation', 'lift', 'lifting', 'movable ink', 're-wettable'],
            '中性-提及兼容铅笔': ['over pencil', 'with pencil',],
            '中性-提及兼容勾线笔': ['over ink', 'with ink', 'over fineliner', 'with fineliner', 'over micron', 'copic-proof'],
            '中性-提及兼容水彩/水粉': ['over watercolor', 'with watercolor', 'with gouache', 'on top of paint'],
            '中性-提及兼容彩铅': ['over colored pencils', 'with colored pencils', 'over pencil crayon', 'with wax pencil'],
            '中性-提及兼容丙烯马克笔': ['on top of acrylic', 'over acrylic', 'with acrylic markers', 'with posca', 'on paint marker'],
        },
        '场景表现': {
            '正面-适合大面积填色': ['great for coloring', 'good for large areas', 'fills spaces evenly', 'no streaking in large blocks', 'coloring book friendly', 'smooth coverage'],
            '负面-不适合大面积填色': ['streaky when coloring', 'dries too fast for large areas', 'bad for filling large spaces', 'leaves marker lines', 'patchy on large areas'],
            '正面-适合漫画/动漫创作': ['great for manga', 'perfect for comics', 'blends skin tones beautifully', 'works for anime style', 'good for cel shading', 'great for character art'],
            '负面-不适合漫画/动漫创作': ['hard to blend skin tones', "colors aren't right for manga", 'smears my line art', 'not good for comic art'],
            '正面-适合插画创作': ['great for illustration', 'professional illustration results', 'layers beautifully for art', 'vibrant illustrations', 'perfect for artists'],
            '负面-不适合插画创作': ['not for professional illustration', 'colors are not vibrant enough for art', 'muddy blends for illustration', 'hobby grade only'],
            '正面-适合着色书/填色': ['great for coloring books', 'perfect for adult coloring', 'coloring book friendly', 'no bleed in coloring book', "doesn't ghost on coloring pages", 'safe for single-sided books', 'fine tip is perfect for intricate designs', 'great for mandalas', 'gets into tiny spaces'],
            '负面-不适合着色书/填色': ['not for coloring books', 'ruined my coloring book', 'bleeds through every page', 'ghosting is too bad for coloring books', 'ruined the next page', 'tip is too broad for detailed coloring', 'bleeds outside the lines in small patterns', 'pills the coloring book paper', 'tears the paper'],
            '正面-适合书法/手写艺术': ['perfect for calligraphy', 'great for hand lettering', 'nice thick and thin strokes', 'good for upstrokes and downstrokes', 'flexible tip for lettering', 'rich black for calligraphy'],
            '负面-不适合书法/手写艺术': ['tip is too stiff for calligraphy', 'hard to control line variation', 'ink feathers during lettering', 'not good for brush lettering', 'ink is not dark enough for calligraphy'],
            '正面-适合手工艺/物品定制': ['great for diy projects', 'perfect for customizing shoes', 'works on canvas bags', 'permanent on rocks and wood', 'good for crafting'],
            '负面-不适合手工艺/物品定制': ['wipes off from plastic', 'not for outdoor use', 'color fades on fabric', "doesn't work on sealed surfaces"],
            '正面-适合儿童/教学': ['great for kids', 'safe for children', 'non-toxic', 'washable ink', 'durable tip for heavy hands', 'bright colors for kids', 'good for classroom use'],
            '负面-不适合儿童/教学': ['strong smell not for kids', 'ink stains clothes', 'tip broke easily with pressure', 'cap is hard for a child to open'],
            '正面-适合刻字/细节': ['perfect for lettering', 'great for calligraphy', 'nice for writing greetings', 'fine tip for small details', 'beautiful for sentiments'],
            '负面-不适合刻字/细节': ['too thick for lettering', 'bleeds when writing', 'hard to do calligraphy with'],
            '正面-多表面DIY': ['perfect for rock painting', 'works great on wood', 'customizing sneakers', 'painting on glass', 'ceramic decorating', 'canvas art', 'outdoor decor', 'painting pumpkins', 'ornament decorating'],
            '负面-不能多表面DIY': ['scrapes off glass', 'not permanent on plastic', 'faded on outdoor rocks', "ink doesn't stick to metal"],
            '中性-提及大面积填色': ['coloring large areas', 'filling in spaces', 'large coverage', 'background coloring'],
            '中性-提及漫画/动漫创作': ['manga', 'comic art', 'anime art', 'line art', 'character art', 'cel shading'],
            '中性-提及插画创作': ['illustration', 'illustrating', 'artwork', 'for my illustrations'],
            '中性-提及着色书/填色': ['coloring book', 'coloring books', 'adult coloring', 'colouring book', 'mandala', 'mandalas', 'intricate designs', 'coloring pages', 'secret garden', 'johanna basford', 'color by number'],
            '中性-提及书法/手写艺术': ['calligraphy', 'hand lettering', 'lettering practice', 'upstrokes', 'downstrokes','typography'],
            '中性-提及手工艺/物品定制': ['diy project', 'craft project', 'crafting with', 'customizing shoes', 'on canvas bags', 'on rocks', 'on wood', 'on plastic', 'on sealed surfaces'],
            '中性-提及儿童/教学': [ 'for kids', 'for children', 'in the classroom', 'for my students', 'art class', 'school project'],
            '中性-提及刻字/细节': ['lettering for cards', 'writing greetings', 'writing sentiments', 'for small details', 'for fine details', 'detailed work'],

        },
        '表面/介质表现': {
            '正面-在纸张上表现好': ['works great on marker paper', 'smooth on bristol board', 'blends well on bleedproof paper', 'perfect for mixed media paper', 'fit for paper', 'good for paper'],
            '负面-在纸张上表现差': ['still bleeds through marker paper', 'feathers on hot press paper', 'destroys bristol surface', 'pills my cold press paper', 'mess up your paper'],
            '中性-提及纸张': ['on paper','marker paper', 'bristol board', 'bristol', 'watercolor paper', 'mixed media paper', 'bleedproof paper', 'hot press', 'cold press', 'sketch book','for paper','for papers'],
            '正面-在深色纸张上显色好': ['opaque on black paper', 'shows up well on dark paper', 'great coverage on kraft paper', 'vibrant on colored paper', 'pops on black', 'shows up beautifully', 'great on black cardstock'],
            '负面-在深色纸张上显色效果差': ['not opaque on black', 'disappears on dark paper', 'too transparent for colored paper', "doesn't show up", 'color looks dull on black'],
            '中性-提及深色纸张': ['black paper', 'dark paper', 'kraft paper', 'colored paper'],
            '正面-在树脂/环氧上表现好': ['works great on resin', 'perfect for resin projects', 'great on epoxy', 'adheres well to resin', 'good on epoxy resin', 'works on epoxy', 'resin art friendly','good for resin crafts', 'resin marker', 'epoxy marker'],
            '负面-在树脂/环氧上表现差': ['doesn\'t work on resin', 'not for resin', 'won\'t stick to resin', 'pools on resin', 'runs on epoxy', 'slides off resin', 'smears on resin','won\'t dry on resin', 'not for resin molds', 'doesn\'t adhere to resin',
                                        'color runs on resin', 'bleeds on epoxy', 'streaky on resin', 'paint pools on epoxy','doesn\'t stick to smooth surfaces', 'won\'t stay on resin', 'runs into puddles',
                                        'not suitable for resin', 'not good on resin', 'useless for resin', 'forms puddles'],
            '正面-在布料上效果好': ['great on fabric', 'permanent on t-shirt', 'holds up in the wash', 'vibrant on textile', 'perfect for customizing shoes', "doesn't feather on cotton", 'survived the wash', 'applies smoothly to canvas', 'flexible on fabric', 'heat sets perfectly', "doesn't stiffen the fabric"],
            '负面-在布料上效果差': ['bleeds on fabric', 'feathers on canvas', 'fades after washing', 'washes out', 'makes the fabric stiff', 'washed right out', 'faded after one wash', 'cracked on the fabric', 'cracks when fabric flexes'],
            '中性-提及布料': ['canvas','canvas mural','on fabric', 'on canvas', 'on t-shirt', 'on textile', 'on cotton', 'on denim', 'for fabric', 'fabric marker'],
            '正面-在木材上表现好': ['great on wood', 'vibrant color on wood', 'dries nicely on wood', 'perfect for wood crafts', 'sharp lines on wood', 'beautiful finish on wood', 'seals nicely', 'vibrant on unfinished wood'],
            '负面-在木材上表现差': ['bleeds into the wood grain', 'color looks dull on wood', 'uneven color on wood', 'smears on sealed wood', 'bleeds with the grain', 'raised the wood grain', 'makes the grain swell'],
            '中性-提及木材': ['on wood', 'for wood', 'writes on wood', 'draw on wood', 'wood grain', 'sealed wood', 'wood crafts', 'unfinished wood'],
            '正面-在石头上表现好': ['great for rock painting', 'vibrant on rocks', 'opaque on stone', 'smooth lines on rocks', 'durable on pebbles', 'covers rocks smoothly', 'perfect for rock art', 'adheres well to stone', 'weather resistant', 'dries quickly on rocks'],
            '负面-在石头上表现差': ['scratches off rocks', 'not opaque enough for stone', 'color is dull on rocks', 'clogs tip on rough stone', 'hard to draw on rocks', 'chips off easily', 'too watery for rocks', 'streaky'],
            '中性-提及石头': ['on rock', 'on rocks', 'on stone', 'on stones', 'on pebble', 'on pebbles', 'for rocks', 'for rock painting', 'rock painting'],
            '正面-在粘土上表现好': ['works on polymer clay', 'great on air dry clay', 'vibrant on clay', 'soaks in nicely on bisque', "doesn't react with sealant", 'adheres perfectly to clay', 'bakes well', 'color stays true after sealing'],
            '负面-在粘土上表现差': ["doesn't adhere to clay", 'smears on polymer clay', 'clogs tip on un-sanded clay', 'reactivates the clay', 'melts the clay surface', 'never fully cures on clay', 'smears easily on polymer clay', 'reacts with glaze'],
            '中性-提及粘土': ['on clay', 'on polymer clay', 'on air dry clay', 'on bisque', 'for clay'],
            '正面-在玻璃(Glass)上表现好': ['permanent on glass', 'smudge proof on glass', 'crisp lines on glass', 'adheres well to glass', 'opaque on glass', 'vibrant on glass', 'writes smoothly on glass', 'removable with windex'],
            '负面-在玻璃(Glass)上表现差': ['wipes off glass', 'smears on glass', 'scratches off glass', 'beads up on glass', 'streaky on glass', 'difficult to remove from glass'],
            '中性-提及玻璃(Glass)': ['on glass', 'for glass', 'writes on glass', 'glass art', 'stain glass'],
            '正面-在陶瓷(Ceramic)上表现好': ['permanent on ceramic', 'writes on mugs', 'decorating ceramic', 'dishwasher safe', 'vibrant on ceramic', 'bake to set', 'cures to a hard finish', 'perfect for customizing mugs', 'great on mugs'],
            '负面-在陶瓷(Ceramic)上表现差': ['never dries on ceramic', 'wipes off ceramic', 'smears on ceramic', 'not dishwasher safe', 'washes off mug', 'scratches off ceramic', 'fades after baking', 'comes right off in dishwasher'],
            '中性-提及陶瓷(Ceramic)': ['on ceramic', 'on mugs', 'on glazed surface', 'for ceramic', 'decorating ceramic'],
            '正面-在塑料(Plastic)上表现好': ['permanent on plastic', 'smudge proof on plastic', 'adheres to plastic', 'vibrant on plastic', 'bonds to plastic', 'dries instantly on plastic', 'great on plastic models'],
            '负面-在塑料(Plastic)上表现差': ['wipes off plastic', 'smears on plastic', "doesn't stick to plastic", 'never dries on plastic', 'rubs off plastic', 'eats the plastic', 'remains sticky on plastic', 'remains tacky'],
            '中性-提及塑料(Plastic)': ['on plastic', 'for plastic', 'writes on plastic', 'plastic models'],
            '正面-在金属(Metal)上表现好': ['adheres to metal', 'permanent on metal', "doesn't scratch off metal", 'clean lines on metal', 'opaque on metal', 'dries quickly on metal', 'marks metal clearly', 'great for metalwork', 'weather resistant on metal'],
            '负面-在金属(Metal)上表现差': ['scratches off metal', 'smears on metal', 'wipes off metal', 'flaked off', 'peeled off metal', 'corrodes metal', 'takes forever to dry on metal', 'rubs off easily', "doesn't adhere to aluminum"],
            '中性-提及金属(Metal)': ['on metal', 'on aluminum', 'for metal', 'marks on metal'],
            '正面-在墙面上表现好': ['great coverage on walls', 'opaque on painted surfaces', 'covers in one coat', 'permanent on drywall', 'durable for murals', 'weatherproof', 'smooth on walls', 'great for mural work', 'low-fume for indoor use'],
            '负面-在墙面上表现差': ['wipes off the wall', 'not for outdoor murals', 'too transparent for walls', 'streaky on walls', 'damaged my wall'],
            '中性-提及墙面': ['on the wall', 'on walls', 'for murals', 'graffiti', 'on drywall', 'on plaster', 'on painted wall'],
        },
        '外观与包装': {
            '正面-外观/设计美观': ['beautiful design', 'minimalist design', 'sleek design', 'clean design', 'well-designed','thoughtful design', 'love the design', 'love the look of', 'pleasing aesthetic', 'looks elegant', 'high-end look', 'modern look', 'looks professional', 'impressed with the design'],
            '负面-外观廉价/丑': ['looks cheap', 'feels cheap', 'cheaply made', 'cheap appearance', 'low-end look', 'plasticky feel', 'flimsy appearance', 'looks like a toy', 'toy-like', 'looks like a child\'s toy','ugly design', 'unattractive design', 'clunky design', 'awkward look', 'poorly designed', 'gaudy colors', 'tacky design', 'looks dated', 'outdated design'],
            '正面-包装美观/保护好': ['very well packaged', 'securely packed', 'well protected','arrived in great shape', 'packaging was good''beautiful packaging', 'nice packaging', 'lovely box', 'great presentation', 'well presented', 'elegant packaging', 'giftable', 'perfect for a gift', 'great gift box', 'nice enough to gift','well packaged', 'packaged securely', 'protective packaging', 'arrived safe', 'arrived in perfect condition', 'no damage during shipping', 'excellent packaging',
                        'sturdy case', 'durable case', 'high-quality box', 'nice tin', 'reusable case', 'great storage tin', 'comes in a nice case'],
            '负面-包装廉价/易损坏': ['flimsy packaging', 'cheap packaging', 'thin cardboard', 'poor quality box', 'doesn\'t protect the pens','damaged box', 'crushed box', 'dented tin', 'arrived damaged', 'damaged in transit', 'damaged during shipping','broken case', 'cracked case', 'case was broken', 'clasp broke', 'latch doesn\'t work', 'zipper broke','cheap case', 'flimsy case', 'case arrived broken'],
            '负面-包装过度':['excessively packaged', 'overly protected', 'too much packaging', 'excessive protection'],
            '正面-收纳便利': ['well-organized', 'keeps them neat', 'keeps them organized', 'easy to organize', 'easy access to colors', 'easy to find the color', 'easy to get pens out', 'convenient storage', 'handy case', 'sturdy case', 'nice carrying case', 'protective case', 'pens fit perfectly', 'individual slots for each pen', 'great storage box', 'useful pen holder'],
            '负面-收纳不便': ['hard to get out', 'difficult to remove pens', 'pens are too tight in the slots', 'struggle to get them out','messy organization', 'poorly organized', 'pens fall out of place', 'don\'t stay in their slots', 'no individual slots', 'pens are all jumbled together', 'hard to put pens back','case doesn\'t close', 'case doesn\'t latch', 'lid won\'t stay closed', 'clasp broke', 'zipper broke','flimsy trays', 'pens fall out when opened'],
            '中性-提及外观': ['pen design', 'overall look', 'visual appeal', 'aesthetic', 'appearance', 'form factor', 'finish', 'color scheme'],
            '中性-提及包装': ['packaging', 'box', 'outer box', 'sleeve', 'tin case', 'gift box', 'presentation', 'protective case', 'blister pack', 'unboxing'],
            '中性-提及收纳': ['storage case', 'carrying case', 'pen holder', 'pen stand', 'pen roll', 'organizer tray', 'layout of the tray', 'how they are organized'],
        },
        '多样性与适配性': {
            '正面-用途广泛': ['multi-purpose', 'all-in-one', 'jack of all trades', 'works for everything','use it for everything', 'handles a variety of tasks', 'works on multiple surfaces',
                      'use on different surfaces', 'good for many different projects', 'one set for all my needs','great for both drawing and writing'],
            '负面-用途单一': ['not versatile', 'lacks versatility', 'not multi-purpose', 'single-purpose', 'single use','one-trick pony', 'limited use', 'very limited in its use', 'limited application',  'only for paper', 'only works on paper', 'doesn\'t work on other surfaces',
                     'only good for one thing', 'useless for anything else', 'very specific use'],
            '正面-可拓展性 (Collection can be expanded)': [ 'expandable collection', 'can add to my collection', 'love adding to my collection', 'complete my collection', 'collect all the colors', 'love that they release new sets', 'new colors available',
                                      'hope they release more colors', 'can\'t wait for new colors','limited edition colors', 'love the special editions', 'collector\'s edition'],
            '负面-可拓展性差 (Poor expandability)': ['no new colors', 'collection is limited', 'wish they had more shades', 'no new sets released','stagnant collection', 'line seems to be discontinued', 'never release new colors',
                                   'can\'t expand my collection', 'no updates to the color range', 'stuck with the same colors', 'wish they would expand the range', 'color range is too small', 'no new releases'],
            '正面-可补充性 (Can be replenished)': ['refillable', 'refillable ink', 'ink refills available', 'can buy refills', 'replaceable cartridges','buy individually', 'can buy single pens', 'sold individually', 'available as singles', 'open stock',
                                 'don\'t have to buy the whole set', 'can just replace the one I need','replaceable nibs', 'can replace the nibs', 'replacement nibs available'],
            '负面-可补充性差 (Poor replenishability)': ["can't buy single", 'not sold individually', 'not available individually', 'can\'t buy individual pens', 'not sold as singles','wish they sold refills', 'no refills available', 'can\'t find refills', 'ink is not refillable', 'no refill cartridges', 'no replacement nibs', 'can\'t replace the tip', 'no replacement parts',
                                   'have to buy a whole new set', 'forced to rebuy the set', 'must buy the entire set again'],
            '正面-单支购买': ['can buy single white pens', 'available as individual markers', 'don\'t need to buy a whole pack for one color', 'sold as singles for replacement'],
            '负面-不可单支购买': ['wasteful to buy a new set', 'no single replacements', 'can\'t find individual pens for sale', 'forced to buy 12 just for the black one'],
            '中性-提及用途广泛性': ['versatility', 'multi-purpose', 'all-in-one', 'works on multiple surfaces', 'use for different things', 'all purpose', 'various uses'],
            '中性-提及可拓展性': ['expandable collection', 'add to the collection', 'complete the set', 'new colors','new sets released', 'limited edition', 'collect all the colors'],
            '中性-提及可补充性': ['refillable', 'open stock', 'sold individually', 'buy single pens', 'replacement nibs', 'ink refills', 'refill cartridges'],
            
            },
        '教育与启发': {
            '正面-激发创意/乐趣': ['fun to use', 'so much fun to play with', 'a joy to use', 'enjoyable to use', 'very satisfying','inspires me to create', 'makes me want to draw', 'makes me want to create', 'sparks my creativity',
                        'boosts my creativity', 'unleashes creativity', 'creative juices are flowing','gets me out of a creative block', 'helps with creative block', 'opens up new possibilities'],
            '正面-适合初学者': ['beginner friendly', 'good for beginners', 'easy for a beginner', 'perfect for beginners','easy to start', 'great starting point', 'just starting out', 'getting started', 'starter kit', 'great starter set', 'my first set', 'new to art', 'new to painting', 'new to drawing', 'first time trying','easy to learn', 'easy to learn with', 'no learning curve', 'no prior experience needed'],
            '负面-有学习门槛': ['steep learning curve', 'learning curve', 'not for beginners', 'not beginner friendly','hard to use', 'difficult to use', 'confusing to use', 'not intuitive', 'hard to control',
                       'difficult to get the hang of', 'takes a lot of practice', 'requires a lot of skill','frustrating for a beginner', 'not easy to get started with'],
            '正面-有教学支持': ['helpful guide', 'clear instructions', 'easy to follow guide', 'step-by-step guide',  'well-written instructions', 'great instruction book','good tutorial', 'helpful video tutorial', 'easy to follow tutorial','great community', 'supportive community', 'helpful facebook group', 'comes with practice sheets', 'love the worksheets', 'great online course'],
            '负面-无教学支持': ['no instructions', 'no guide included', 'didn\'t come with instructions', 'no user manual', 'lacks instructions', 'confusing guide', 'unhelpful guide', 'hard to understand instructions', 'instructions are not clear', 'useless instructions', 'poorly written', 'vague instructions', 'bad translation','instructions in another language', 'only in chinese',
                       'no online tutorials', 'can\'t find any videos on how to use'],
            '中性-提及创意/乐趣': [ 'creative juices', 'fun activity', 'joy of creating', 'spark creativity', 'boost creativity','creative outlet', 'artistic expression', 'fun to use', 'enjoyable process', 'doodling for fun'],
            '中性-提及学习门槛': ['beginner friendly', 'good for beginners', 'easy for a beginner','starter kit', 'starter set', 'my first set', 'entry-level','learning curve', 'no prior experience', 'easy to learn with',
                        'just starting out', 'getting started','new to art', 'new to painting', 'new to drawing','learning to draw', 'learning to paint'],
            '中性-提及教学支持': ['instruction book', 'instructional booklet', 'guidebook', 'step-by-step guide', 'how-to guide', 'learning guide', 'video tutorial', 'youtube tutorial', 'following a tutorial',
                        'online course', 'skillshare class', 'practice sheets', 'worksheets','online community', 'facebook group'],
        },
        '特殊用途': {
            '正面-专业级表现': ['professional grade', 'artist grade', 'pro grade', 'professional quality', 'artist quality', 'studio grade', 'museum quality', 'for serious artists', 'not student grade','professional results', 'gallery quality results', 'publication quality',
                       'industry standard', 'lightfast', 'excellent lightfastness', 'high lightfastness rating', 'fade-resistant', 'fade proof', 'archival quality', 'archival ink', 'archival pigment'],
            '负面-非专业级': ['not professional grade', 'not artist grade', 'hobby grade', 'student grade', 'for hobby use only', 'for casual use only', 'not for serious artists',
                      'feels like a toy', 'not for client work', 'not for commissions','not archival', 'not lightfast', 'more of a toy than a tool'],
            '中性-提及专业性': ['professional grade', 'artist grade', 'hobby grade', 'student grade', 'pro grade', 'lightfast', 'lightfastness rating', 'archival quality', 'archival ink', 'museum quality'],
        },
        '性价比': {
            '正面-性价比高': ['affordable', 'cheap', 'good value', 'great deal', 'worth the money', 'great buy', 'reasonable price', 'cheaper than', 'alternative to','excellent value', 'amazing value','inexpensive','low price', 'great price point','money well spent', 'can\'t beat the price'],
            '负面-价格昂贵': ['expensive', 'overpriced', 'not worth', 'pricey', 'costly', 'rip off', 'too much', 'waste of money','not worth it','over-priced'],

            },
        '配套与服务(色卡)': {
            '正面-提供色卡/好用': ['comes with a swatch card', 'includes a swatch card', 'love the swatch card', 'helpful swatch card', 'great for swatching', 'easy to swatch', 'blank swatch card', 'pre-printed swatch card'],
            '负面-缺少色卡/不好用': ['no swatch card', "wish it had a swatch card", "doesn't come with a swatch card", 'had to make my own swatch card', 'swatch card is inaccurate', 'swatch card is useless', "colors on swatch card don't match"],
            '中性-提及色卡': ['swatch card',  'color chart'],
            },
        '购买与服务体验': {
            '正面-开箱/展示': ['beautiful presentation', 'great unboxing experience', 'perfect for a gift', 'looks professional', 'elegant packaging', 'giftable', 'nice gift box', 'well presented', 'impressive presentation',
                       'lovely box', 'makes a great gift', 'nicely laid out'],
            '负面-运输/损坏': ['damaged on arrival', 'missing marker', 'pen was missing', 'not all included', 'empty slot', 'marker missing', 'one less than expected', 'incomplete set',
                      'arrived broken', 'arrived broken', 'pens arrived broken', 'some were broken', 'cracked on arrival', 'damaged during shipping','damaged in transit', 'arrived damaged', 'item was damaged','leaking ink', 'leaked all over', 'ink leaked everywhere', 'arrived leaking','box was crushed',
                      'package was damaged', 'box was open', 'dented tin', 'poorly packaged for shipping', 'not well protected', 'arrived in bad shape'],
            '正面-客服/售后': ['great customer service', 'excellent customer service', 'amazing support', 'seller was helpful', 'seller was very helpful', 'very responsive seller', 'quick response', 'fast reply',
                      'answered my questions quickly', 'resolved my issue quickly', 'problem solved','fast replacement', 'quick replacement', 'sent a replacement right away', 'easy replacement process',
                       'easy refund', 'hassle-free refund', 'full refund was issued','went above and beyond', 'proactive customer service'],
            '负面-客服/售后': ['bad customer service', 'terrible customer service', 'poor support', 'no customer service','seller was unresponsive', 'no response from seller', 'never replied', 'took forever to respond', 'slow response',
                      'seller was unhelpful', 'refused to help', 'unwilling to help', 'could not resolve the issue','missing items', 'missing parts', 'didn\'t receive all items',
                       'wrong item sent', 'received the wrong color', 'sent the wrong size','difficult return process', 'hassle to get a refund', 'refused a refund', 'no replacement offered'],
            '中性-提及开箱/展示': ['unboxing experience', 'presentation', 'packaging', 'giftable', 'nice box', 'sturdy case', 'storage tin', 'well organized', 'comes in a case'],
            '中性-提及运输': ['shipping', 'delivery', 'arrival condition', 'transit', 'shipped', 'arrived',  'damage','damaged', 'broken', 'crushed', 'shipping box', 'protective packaging'],
            '中性-提及客服/售后': ['customer service', 'contacted seller', 'contacted support', 'seller response','replacement', 'refund', 'return process', 'exchange', 'missing items', 'wrong item sent', 'issue resolved'],
        }
}

# --- 2. 产品分类映射 (Product & Category Mapping) ---
USER_CATEGORY_MAPPING = {
    'B001AS6P4G': 'B001AS6P4G_posca_(1.85-3.33)_阀门式_8mm_Chisel_15_纸盒_插口盒',
    'B00ZC9JK4Q': 'B00ZC9JK4Q_posca_(1.85-3.33)_阀门式_1.8-2.5mm_Medium_7_纸盒_插口盒',
    'B013NMZ92U': 'B013NMZ92U_posca_(>29.97)_阀门式_0.9-1.3mm_Fine_1_PET插口盒_插口盒',
    'B06ZYTZXG2': 'B06ZYTZXG2_posca_(1.85-3.33)_阀门式_0.9-1.3mm_Fine_24_纸盒_天地盖硬纸盒',
    'B07D2LC8LH': 'B07D2LC8LH_ARTISTRO_(1.17-1.85)_阀门式_0.7mm_Extra Fine_5_纸盒_插口盒',
    'B07NRW6H8M': 'B07NRW6H8M_ARTISTRO_(3.33-29.97)_阀门式_0.7mm_Extra Fine_1_纸盒_插口盒',
    'B086XCN9CR': 'B086XCN9CR_AKARUED_(1.17-1.85)_阀门式_2-3mm_Medium_8_PET插口盒_插口盒',
    'B08LTYXQLC': 'B08LTYXQLC_ArtShip Design_(3.33-29.97)_阀门式_0.7-2mm_Extra Fine&Medium_2_纸盒_插口盒',
    'B098QPHCVT': 'B098QPHCVT_AKARUED_(1.17-1.85)_阀门式_0.7mm_Extra Fine_8_纸盒_插口盒',
    'B09SWJRSSX': 'B09SWJRSSX_JR.WHITE_(0.53-0.79)_阀门式_0.7mm_Extra Fine_18_纸盒_插口盒',
    'B09TR9SD6P': 'B09TR9SD6P_SAKEYR_(1.17-1.85)_阀门式_2-3mm_Medium_12_纸盒_插口盒',
    'B09VCYS41G': 'B09VCYS41G_Betem_(0.46-0.53)_棉芯式_1-5mm_Fine&Dot_24_纸盒_插口盒',
    'B0B7VZ9KTJ': 'B0B7VZ9KTJ_Artugn_(0.53-0.79)_棉芯式_0.5-5mm_Brush&Fine_24_纸盒_插口盒',
    'B0BN8CW9ZY': 'B0BN8CW9ZY_TANMIT_(0.53-0.79)_棉芯式_1-3mm_Fine&Dot_24_牛津布_布袋',
    'B0C2L17JZB': 'B0C2L17JZB_Tesquio_(0.79-1.17)_棉芯式_0.5-5mm_Brush&Dot_8_纸盒_插口盒',
    'B0C61HTPV9': 'B0C61HTPV9_SFAIH_(0.53-0.79)_阀门式_3mm_Medium_24_PET插口盒_插口盒',
    'B0CDL915XY': 'B0CDL915XY_SFAIH_(0.79-1.17)_阀门式_0.7-3mm_Extra Fine&Medium_8_纸盒_插口盒',
    'B0CKN74CW3': 'B0CKN74CW3_BIGTHUMB_(3.33-29.97)_阀门式_3-15mm_Jumbo_3_纸盒_插口盒',
    'B0CNYV8724': 'B0CNYV8724_TFIVE_(3.33-29.97)_阀门式_0.7mm_Extra Fine_2_纸盒_插口盒',
    'B0CP48JPCH': 'B0CP48JPCH_ARTISTRO_(0.46-0.53)_棉芯式_1-5mm_Fine&Dot_24_纸盒_插口盒',
    'B0CQL15B98': 'B0CQL15B98_SFAIH_(0.53-0.79)_阀门式_3mm_Medium_36_PET插口盒_插口盒',
    'B0CX1BD86P': 'B0CX1BD86P_TBC The Best Crafts_(<0.37)_棉芯式_1-3.9mm_Brush&Dot_24_纸盒_插口盒',
    'B0CXM7VLXV': 'B0CXM7VLXV_Droaful_(<0.37)_棉芯式_1-5mm_Brush&Fine_20_纸盒_插口盒',
    'B0CYCCR82G': 'B0CYCCR82G_SRUOLOC_(1.85-3.33)_阀门式_3-15mm_Jumbo_3_纸盒_插口盒',
    'B0CYSZ52T4': 'B0CYSZ52T4_SRUOLOC_(3.33-29.97)_阀门式_3-15mm_Jumbo_1_纸盒_插口盒',
    'B0D21NHYX8': 'B0D21NHYX8_SFAIH_(1.17-1.85)_阀门式_2-3mm_Medium_4_纸盒_插口盒',
    'B0D2H7QSX2': 'B0D2H7QSX2_Coogert_(0.46-0.53)_棉芯式_0.5-5mm_Brush&Fine_30_纸盒_插口盒',
    'B0D471YK2F': 'B0D471YK2F_Oficrafted_(<0.37)_棉芯式_0.3mm_Medium_80_牛津布_拉链式多层布袋',
    'B0D8NGYP76': 'B0D8NGYP76_Artecho_(0.42-0.46)_棉芯式_0.5-5mm_Brush&Fine_48_纸盒_插口盒',
    'B0DFMRDMNJ': 'B0DFMRDMNJ_REALZEVA_(<0.37)_阀门式_1-3mm_Fine&Dot_24_纸盒_插口盒',
    'B0DJBS38PR': 'B0DJBS38PR_Amamao_(0.46-0.53)_直液式_1-5mm_Brush_120_纸盒_天地盖硬纸盒',
    'B0DKFMN8HF': 'B0DKFMN8HF_AKARUED_(>29.97)_棉芯式_1mm_Brush&Fine_6_纸盒_插口盒',
    'B0DL9ZPMLX': 'B0DL9ZPMLX_Recheel_(<0.37)_棉芯式_1-5mm_Brush&Fine_30_纸盒_插口盒',
    'B0DLKVD4TW': 'B0DLKVD4TW_nauets_(0.53-0.79)_棉芯式_1-5mm_Fine&Dot_30_纸塑结合包装_插口盒',
    'B0DMNY5C69': 'B0DMNY5C69_TOSHARE_(>29.97)_棉芯式_1-5mm_Brush&Fine_30_纸盒_插口盒',
    'B0DQX43YQD': 'B0DQX43YQD_TBC The Best Crafts_(>29.97)_棉芯式_1-3.9mm_Brush&Dot_1_牛津布_拉链式多层布袋',
    'B0DRFG52NK': 'B0DRFG52NK_IVSUN_(0.53-0.79)_棉芯式_0.5-5mm_Brush&Fine_30_纸盒_插口盒',
    'B0DS1NZD89': 'B0DS1NZD89_HOTU_(>29.97)_直液式_0.7-6mm_Brush_36_纸盒_天地盖硬纸盒',
    'B0F5WMP62Z': 'B0F5WMP62Z_Coogert_(<0.37)_棉芯式_1-5mm_Fine&Dot_24_纸盒_插口盒',
    'B0F66RVHJW': 'B0F66RVHJW_Overseas_(>29.97)_直液式_0.1-5mm_Brush_60_纸盒_插口盒'
}

# --- 3. 消费者用户画像 ---
CLASSIFICATION_RULES = {
        "User_Role": {
          '专业艺术工作者 (Professional Artist)': ['professional artist', 'pro artist', 'artist', 'illustrator', 'designer', 'comic artist', 'manga artist', 'architect', 'art studio', 'in my studio', 'commission', 'client work', 'freelance artist', 'professional work'],
          '学生 (Student)': ['student', 'school', 'college', 'university', 'art student', 'design student', 'for class', 'in my class','class notes', 'study notes', 'school project', 'assignment', 'textbook', 'studying for'],
          '教师 (Teacher)': ['teacher', 'educator', 'professor', 'art teacher', 'instructor', 'for my classroom','teaching a class', 'grading papers', 'school supplies for my students'],
          '父母 (Parent)': ['parent', 'mom', 'dad', 'mother', 'father', 'for my kids', 'for my son', 'for my daughter', 'family craft', 'homeschooling', 'with my kids'],
          '手账爱好者 (Journaler/Planner)': ['journaling', 'in my journal', 'art journal', 'junk journal', 'bible journaling','decorating my planner', 'in my planner',
                             'setting up my planner', 'planner decoration', 'planner stickers', 'planner layout','bujo', 'bullet journal',
                             'bujo spread','in my diary','scrapbooking', 'scrapbook layout', 'making a scrapbook','memory keeping', 'memory planner',
                             'hobonichi', 'leuchtturm', 'traveler\'s notebook','washi tape', 'journal stickers'
                            ],
          '业余艺术爱好者 (Hobbyist)': ['hobbyist', 'amateur artist', 'as a hobby', 'just a hobby', 'passion project', 'in my spare time', 'self-taught artist', 'just for fun', 'drawing for fun',  'painting for fun', 'my favorite pastime', 'weekend artist', 'doodling for fun','sketching in my free time', 'not a professional', 'not an artist but', 'art as a hobby'],
          '文化创意从业者 (Creative Professional)': ['creative professional', 'artisan', 'craft market', 'craft fair', 'artisan market','etsy seller', 'selling on etsy',
                                 'my etsy shop', 'small business owner', 'selling my art','content creator', 'youtuber', 'instagram artist', 'patreon creator', 'twitch streamer', 'art blogger',
                                 'workshop host', 'skillshare teacher', 'art instructor', 'leading a workshop'
                                 ],
          '初学者 (Beginner)': ['beginner', 'newbie', 'novice', 'beginner friendly', 'good for beginners', 'easy for a beginner','just starting', 'just starting out', 'getting started', 'great starting point',
                       'starter kit', 'starter set', 'my first set',  'new to art', 'new to painting', 'new to drawing', 'first time trying',
                      'learning to draw', 'learning to paint', 'easy to learn with', 'no prior experience'
                      ],          '办公人士 (Business/Office Professional)': ['for the office', 'at my office', 'office supplies', 'office work', 'at work', 'for my business','business meeting', 'work presentation', 'meeting notes', 'mind mapping for work', 'whiteboard at work',
                                  'corporate training', 'coworker', 'official report', 'signing documents', 'desk organization', 'organizing my desk'
                                 ],
          '艺术疗愈/健康追求者 (Art Therapy/Wellness Seeker)': ['art therapy', 'therapeutic', 'for relaxation', 'to relax', 'calming activity', 'for mindfulness',
                                       'helps with my anxiety', 'stress relief', 'to unwind', 'for my mental health', 'self-care activity',
                                        'peaceful activity', 'meditative drawing', 'helps me escape', 'clears my head', 'zone out',
                                        'calms me down', 'relaxing hobby', 'de-stress'
                                      ],
          '机构/批量采购者 (Institutional/Bulk Purchaser)': ['bulk order', 'bulk purchase', 'large order', 'large quantity',  'for the whole class', 'for my classroom', 'classroom set', 'school supplies order','for the office', 'office supply order', 'stocking the office', 'office set','church group', 'for the church', 'community center', 'non-profit', 'for our team', 'event supplies', 'charity donation', 'donation for',
                                      'stock up for the office', 'stock up for the classroom'],
          },
        "Gender": {
            '女性 (Female)': ['woman', 'women', 'girl', 'girls', 'she','niece','her','hers', 'wife', 'mother', 'mom', 'daughter', 'girlfriend', 'female', 'sister', 'aunt', 'grandmother', 'niece', 'lady', 'ladies'],
            '男性 (Male)': ['man', 'men', 'nephew','boy', 'boys', 'he', 'his', 'him', 'husband', 'father', 'dad', 'son', 'boyfriend', 'male', 'brother', 'uncle', 'grandfather', 'nephew', 'gentleman']
        },
        "Age_Group": {
            '儿童/幼儿': ['kid', 'kids', 'child', 'children', 'toddler', 'baby', 'preschooler', 'little one', 'grandson', 'granddaughter', 'for my son', 'for my daughter','nephew', 'niece', 'elementary school'],
            '青少年/学生': ['teen', 'teenager', 'adolescent', 'youth', 'high school', 'middle school', 'college student', 'university student', 'art student', 'for class'],
            '成年人/专业人士': ['adult', 'professional', 'pro artist', 'office work', 'at work', 'client work', 'in my studio', 'career', 'adult coloring'],
            '老年人': ['senior', 'elderly', 'retired', 'grandparent', 'grandfather', 'grandmother', 'golden years', 'grandma', 'grandpa']
        },
        "Usage": {
            '填色本填色 (Coloring Book)': ['coloring book', 'coloring books', 'adult coloring', 'colouring book', 'color page', 'coloring pages',  'adult coloring book', 'color therapy', 'mindfulness coloring', 'relaxing coloring', 'intricate coloring', 'detailed coloring', 'secret garden', 'johanna basford', 'kerby rosanes', 'hanna karlzon', 'mandalas', 'mandala coloring', 'color by number', 'mystery coloring'],
            '书法与手写艺术 (Calligraphy & Hand Lettering)': ['calligraphy', 'lettering', 'hand lettering', 'typography', 'modern calligraphy', 'brush lettering', 'faux calligraphy', 'handlettering', 'scripting', 'pointed pen', 'envelope addressing', 'flourishing', 'copperplate script', 'spencerian script', 'journal headers', 'planner headers',  'writing letters', 'place cards', 'wedding invitations'],
            '绘画创作 (Art Creation)': ['making art', 'creating art', 'for my art', 'art project', 'fine art', 'for a drawing', 'for drawing',  'illustration', 'manga', 'comic art',
                            'landscape sketch', 'urban sketching', 'artwork', 'portrait drawing',  'character design', 'sketching', 'botanical illustration', 'still life', 'figure drawing',
                            'plein air painting', 'doodling for art', 'zentangle art', 'watercolor painting', 'acrylic painting', 'inking lines', 'animal drawing', 'concept art'
            ],
            '设计工作 (Design Work)': ['design work', 'for my design work', 'professional design', 'client design', 'design project','fashion design', 'fashion illustration', 'garment design', 'textile design',
                           'product design', 'industrial design', 'product sketch', 'rendering', 'graphic design', 'logo design', 'layout design', 'branding', 'ui design', 'ux design', 'wireframing', 'mockup','architecture',
                          'architectural drawing', 'interior design', 'floor plan', 'blueprint', 'schematics','concept art', 'character design', 'storyboard', 'set design'
            ],
            '教学与学习 (Teaching & Learning)': ['art class', 'craft class', 'art school', 'for my students', 'for the class', 'classroom supplies','student work', 'school project', 'class assignment', 'grading papers', 'lesson planning', 'teaching a class', 'art education', 'homeschooling', 'learning to draw', 'learning to paint', 'skillshare class', 'online course', 'art tutorial',  'following a tutorial', 'art demonstration'],
            '手账装饰 (Journal & Planner Decoration)': ['note taking', 'taking notes', 'study notes', 'meeting notes', 'class notes', 'annotating books',  'marking up documents', 'color coding', 'color code my notes', 'organizing my calendar', 'calendar planning', 'labeling', 'making labels', 'organizing files', 'to-do list',
                                     'making lists', 'grocery list', 'keeping track of'],
            '日常记录与组织 (Daily Organization)': ['note taking', 'taking notes', 'study notes', 'meeting notes', 'class notes', 'annotating books',  'marking up documents', 'color coding', 'color code my notes', 'organizing my calendar', 'calendar planning', 'labeling', 'making labels', 'organizing files', 'to-do list',
                                  'making lists', 'grocery list', 'keeping track of'],
            '卡片与礼品制作 (Card & Gift Making)': ['card making', 'greeting card', 'handmade card', 'gift tag', 'decorating gifts',  'making cards', 'birthday card', 'christmas cards', 'thank you card', 'thank you notes', 'wedding invitations', 'anniversary card', 'valentines card', 'personalizing gifts',  'custom gifts', 'wrapping paper', 'gift wrap', 'envelope addressing', 'sentiments for cards'],
            '儿童涂鸦与早教 (Kids Activities)': ['for my kids', 'for the kids', 'with my children', 'for my toddler', 'for my son', 'for my daughter', 'kids craft', 'crafts for kids', 'family craft time', 'family fun', 'art project for kids', 'doodling', 'scribbling', 'finger painting', 'mess-free coloring','early learning', 'educational toy', 'learning colors', 'develop fine motor skills', 'preschool activities',
                               'safe for children', 'kid friendly', 'rainy day activity'],
            'DIY与手工制作 (DIY & Crafts)': ['diy project', 'craft project', 'crafting', 'for crafts', 'arts and crafts', 'handmade gifts','decorating ornaments', 'customizing shoes', 'phone case decoration', 'painting pumpkins', 'easter egg decorating', 'on glass', 'on t-shirt', 'on fabric', 'on plastic', 'on metal', 'model painting', 'miniature painting', 'painting miniatures', 'warhammer painting', 'model building',  'customizing', 'rock painting', 'mug decoration', 'wood burning', 'wood signs', 'wood crafts', 'resin art', 'resin crafts', 'polymer clay crafts', 'jewelry making', 'candle making', 'wreath making'],
            '户外与旅行创作 (Outdoor & Travel Art)': ['outdoor drawing', 'en plein air', 'urban sketching', 'travel journal', 'traveling with', 'on the go', 'field sketch','portable for travel'],
            '收藏与展示 (Collection & Display)': ['add to my collection', 'complete my collection', 'collector', 'collector\'s item', 'limited edition', 'special edition', 'collectible set', 'complete the set', 'the whole set', 'full set', 'for display', 'on my shelf'],
            '文化体验与活动 (Cultural Activities)': ['workshop', 'art event', 'cultural festival', 'live drawing', 'art therapy session', 'community art'],
            '心理疗愈 (Therapeutic Use)': ['for relaxing', 'for relaxation', 'stress relief', 'art therapy', 'therapeutic', 'calming', 'for mindfulness','emotional outlet', 'doodling to relax', 'zen', 'to unwind']
        },
        "Motivation": {
            '专业需求-色彩表现': ['high quality pigment', 'high pigment load', 'richly pigmented', 'pure pigment', 'vibrant colors', 'rich colors', 'deep saturation', 'consistent saturation', 'intense colors','lightfast', 'excellent lightfastness', 'lightfastness rating', 'archival quality', 'archival ink', 'museum quality','smooth blending', 'blends seamlessly', 'layering without getting muddy', 'excellent blendability', 'good for glazing', 'lifts cleanly', 'non-staining', 'good staining properties','true to color', 'color accuracy', 'good opacity', 'opaque coverage', 'good transparency'],
            '专业需求-性能耐用': ['pro grade', 'professional grade', 'reliable for work', 'consistent flow', 'consistent performance', 'durable tip', 'long lasting', 'for professional work', 'serious tool', 'heavy duty', 'withstand pressure', 'workhorse', 'built to last', 'daily driver', 'holds up to heavy use',  'no skipping', 'dependable performance', 'withstands abuse', 'for demanding work'],
            '基础功能需求': ['for basic use', 'for everyday use', 'for daily use', 'for school', 'for taking notes', 'gets the job done', 'does the job', 'all i need', 'nothing fancy', 'just the basics', 'no frills', 'simple and effective', 'standard use', 'for general use'],
            '艺术兴趣驱动': ['for my hobby', 'passion for art', 'spark creativity', 'express myself', 'for fun', 'artistic exploration','wanted to try', 'get back into art','for hobby'],
            '品牌信任': [ 'trusted brand', 'good reputation', 'well-known brand', 'always reliable', 'go-to brand', 'love this brand','stick with this brand', 'brand loyalty',],
            '性价比驱动': [ 'good value', 'great price', 'affordable', 'on a budget', 'good deal', 'cheap but good', 'cost effective', 'cheaper alternative'],
            '创新功能吸引': ['innovative feature', 'new feature',  'unique feature', 'special feature','new technology'],   
            '外观设计吸引': ['love the design', 'beautiful aesthetic', 'looks good', 'pretty colors', 'minimalist design', 'the look of it', 'elegant design'],
            '包装与开箱体验吸引': ['beautiful packaging', 'great unboxing experience', 'giftable', 'nice box', 'good presentation'],
            '社交驱动-口碑推荐': ['recommendation', 'recommended by', 'my friend recommended', 'my teacher recommended', 'word of mouth', 'told me to buy','saw good reviews'],
            '社交驱动-媒体影响': ['saw it on social media', 'tiktok made me buy it', 'saw it on instagram', 'youtube review', 'influencer recommended'],
            '文化与身份认同': ['culture', 'themed set', 'limited edition', 'collaboration', 'artist series', 'collectible', 'part of my identity'],
            '便携性需求': ['portable', 'on the go', 'easy to carry', 'travel set', 'compact', 'lightweight','great for travel', 'perfect for travel', 'traveling with', 'take it anywhere',  'fits in my bag', 'fits in my pocket', 'small size', 'doesn\'t take up much space','comes with a travel case', 'nice travel case'],
            '多功能性需求': ['multi-purpose', 'all-in-one', 'jack of all trades', 'works for everything', 'use it for everything', 'handles a variety of tasks', 'works on multiple surfaces','good for many things', 'one set for all my needs','many surface'],
            '礼品需求': [ 'as a gift', 'for a present', 'gift for someone', 'birthday gift', 'christmas gift', 'holiday gift','for gifts','for gift'],
            '激发创造力': ['spark creativity', 'boost creativity', 'unleash creativity', 'explore my creativity', 'creative block', 'helps with creative block', 'overcome creative block', 'new ideas',  'get the creative juices flowing', 'makes me want to create', 'inspires me to create'],
            '缓解压力与情绪调节': ['stress relief', 'for relaxation', 'to relax', 'calming', 'art therapy', 'therapeutic', 'for mindfulness', 'to unwind', 'zone out', 'peaceful activity', 'meditative', 'escape from reality', 'helps with my anxiety', 'calms me down', 'relaxing hobby'],
            '满足好奇心': ['curious about', 'wanted to see', 'heard about', 'first impression', 'wanted to try',  'give it a try', 'out of curiosity', 'just to see', 'intrigued by', 'see what the hype is about', 'first time trying', 'wanted to check it out'],
            '环保与可持续性': ['eco-friendly', 'sustainable', 'made from recycled materials', 'recycled plastic', 'recyclable', 'biodegradable', 'refillable', 'non-toxic', 'less waste', 'zero waste', 'earth-friendly',  'planet-friendly', 'good for the environment', 'environmentally friendly', 'plant-based ink', 'recyclable packaging'],
            '支持特定文化': ['support local artist', 'support local brand', 'made by local artist', 'cultural collaboration',  'artist collaboration', 'support small business',  'national pride', 'traditional craft', 'heritage brand', 'artist series', 'indie brand'],
            '追随潮流': ['trending on social media', 'all the hype', 'everyone has it', 'in style', 'viral', 'saw it on tiktok','tiktok','tiktok made me buy it', 'instagram made me buy it','instagram', 'all over youtube', 'influencer recommended', 'all the rage', 'latest trend', 'hyped up product'],
            '效率驱动': ['more efficient', 'improves efficiency', 'quick drying', 'fast drying', 'dries instantly', 'fast-drying',  'saves me time', 'saves time', 'work faster', 'speeds up my process', 'improves my workflow', 'streamline my workflow', 'cuts down on time', 'hassle-free', 'get the job done faster'],
            '学习新技能': ['learning a new skill', 'good for tutorials', 'starter kit', 'for beginners', 'beginner friendly',  'easy to learn', 'learning to draw',
                    'learning calligraphy', 'want to learn', 'just starting out',  'new to art', 'first time trying', 'for practice', 'practicing my skills',
                    'comes with instructions',  'step-by-step guide', 'improve my drawing', 'get better at painting'],
            '提升现有技能': ['challenge myself', 'advanced techniques','better tool', 'step up my game','refine my skills', 'take my art to the next level', 'more control over lines','fine-tune my work', 'mastering the craft',  'invest in my art', 'for advanced users', 'expand my capabilities', 'unlock new techniques']
        }
}

# --- 3. 数据加载函数 (已修改：统一为单一文件入口) ---
@st.cache_data
def load_raw_data():
    # === 修改核心：只读取一个名为 '常青款.xlsx' 的文件，不再区分销量和趋势 ===
    # 这里的 key 是文件名，value 是 (主类目, 子类目名称)
    # 子类目统一命名为 "全量数据"，避免页面出现分类
    data_map = {
        "常青款.xlsx": ("常青款", "全量数据")
    }
    
    combined = []
    for filename, info in data_map.items():
        if os.path.exists(filename):
            df_temp = pd.read_excel(filename)
            
            # --- 自动寻找列名 ---
            col_name = next((c for c in ['Content', 'Review Body', 'Body', 'content'] if c in df_temp.columns), df_temp.columns[0])
            asin_col = next((c for c in ['ASIN', 'Parent ASIN', 'Product ID', 'Asin', 'child_asin'] if c in df_temp.columns), None)
            
            # --- 映射逻辑 ---
            if asin_col:
                df_temp['sku_spec'] = df_temp[asin_col].astype(str).str.strip().str.upper().map(USER_CATEGORY_MAPPING).fillna("Other-Unmapped")
            else:
                df_temp['sku_spec'] = "Unknown-Spec"

            # --- 句子拆分与情感分析 ---
            df_temp = df_temp.dropna(subset=[col_name])
            
            def split_and_analyze(text):
                sentences = sent_tokenize(str(text).lower())
                results = []
                for s in sentences:
                    pol = TextBlob(s).sentiment.polarity
                    results.append({'text': s, 'polarity': pol})
                return results

            df_temp['sentences'] = df_temp[col_name].apply(split_and_analyze)
            df_exploded = df_temp.explode('sentences')
            
            # 安全提取句子内容
            df_exploded['s_text'] = df_exploded['sentences'].apply(lambda x: x['text'] if isinstance(x, dict) else "")
            df_exploded['s_pol'] = df_exploded['sentences'].apply(lambda x: x['polarity'] if isinstance(x, dict) else 0)
            
            df_exploded['main_category'] = info[0]
            df_exploded['sub_type'] = info[1]
            
            combined.append(df_exploded)
    
    return pd.concat(combined, ignore_index=True) if combined else pd.DataFrame()
    

# --- 4. 核心分析逻辑 (保持不变) ---
def extract_advanced_features(df):
    """为每一句评论打上画像、场景、动机标签"""
    processed_df = df.copy()
    for dim_name, sub_dict in CLASSIFICATION_RULES.items():
        clean_col_name = "feat_" + dim_name
        def get_tag(text):
            text_lower = str(text).lower()
            for tag, keywords in sub_dict.items():
                if any(str(k).lower() in text_lower for k in keywords):
                    return tag
            return "未提及"
        processed_df[clean_col_name] = processed_df['s_text'].apply(get_tag)
    return processed_df


def analyze_sentiments(df_sub):
    results = []
    total_reviews_count = len(df_sub)
    
    for category, sub_dict in FEATURE_DIC.items():
        pos_score, neg_score = 0.0, 0.0
        hit_details, matched_ratings = [], []
        dimension_vocal_count = 0 

        for tag, keywords in sub_dict.items():
            pattern = '|'.join([re.escape(k) for k in keywords])
            mask = df_sub['s_text'].str.contains(pattern, na=False)
            matched_df = df_sub[mask]
            
            if not matched_df.empty:
                if '负面' in tag:
                    valid_match = matched_df[(matched_df['s_pol'] < 0.1) | (matched_df['Rating'] <= 3)]
                    weight = 1.5 if valid_match['Rating'].mean() <= 2.1 else 1.0
                    neg_score += (len(valid_match) * weight)
                    count = len(valid_match)
                elif '正面' in tag:
                    valid_match = matched_df[(matched_df['s_pol'] > -0.1) | (matched_df['Rating'] >= 4)]
                    pos_score += len(valid_match)
                    count = len(valid_match)
                else: count = 0

                if count > 0:
                    dimension_vocal_count += count
                    matched_ratings.extend(valid_match['Rating'].tolist())
                    if '负面' in tag:
                        hit_details.append(f"{tag.split('-')[-1]}({count}次)")

        dim_rating = np.mean(matched_ratings) if matched_ratings else 0
        total_vocal = pos_score + neg_score
        sentiment_score = (pos_score / total_vocal * 100) if total_vocal > 0 else 0
        
        # 置信度缩放
        confidence = np.log1p(dimension_vocal_count) / np.log1p(max(total_reviews_count/5, 1)) 
        confidence = min(max(confidence, 0.5), 1.2)
        
        opp_index = round((neg_score * (100 - sentiment_score) * (5.1 - dim_rating) / 100) * confidence, 2)

        results.append({
            "维度": category, "亮点": int(pos_score), "痛点": int(neg_score),
            "满意度": round(sentiment_score, 1), "维度评分": round(dim_rating, 2),
            "机会指数": opp_index, "痛点分布": ", ".join(hit_details) if hit_details else "无"
        })
    return pd.DataFrame(results)

# --- 5. Streamlit 页面布局 ---
st.set_page_config(page_title="丙烯笔深度调研", layout="wide")
st.title("🎨 丙烯马克笔消费者洞察看板")

df = load_raw_data()

if not df.empty:
    # 侧边栏 (由于只有一个类目，这里其实只是做个筛选的样子，实际只有一个选项)
    target = st.sidebar.radio("🎯 选择分析类目", df['main_category'].unique())
    filtered = df[df['main_category'] == target]
    
    # 此时 sub_types 应该只有一个值 ["全量数据"]
    sub_types = filtered['sub_type'].unique()

    # 遍历子类型 (这里只会循环一次)
    for sub_name in sub_types:
        # 为了美观，如果只有一种数据，可以去掉顶部的分隔空白，或者保留
        st.write("") 
        
        # 头部标题块
        st.markdown(f"""
            <div style="
                background-color: #f8f9fa; 
                padding: 20px; 
                border-radius: 15px; 
                margin-top: 20px; 
                margin-bottom: 30px; 
                border-left: 10px solid #1f77b4;
                box-shadow: 2px 2px 5px rgba(0,0,0,0.05);
            ">
                <h2 style="
                    margin: 0; 
                    color: #1f77b4; 
                    font-size: 36px; 
                    font-weight: bold;
                ">
                    常青款深度洞察
                </h2>
            </div>
        """, unsafe_allow_html=True)
        
        sub_df = filtered[filtered['sub_type'] == sub_name]
        analysis_res = analyze_sentiments(sub_df)
        
        # 顶部指标卡
        m1, m2, m3, m4 = st.columns(4)
        total_pos = analysis_res["亮点"].sum()
        total_neg = analysis_res["痛点"].sum()
        health_rate = round(total_pos / (total_pos + total_neg) * 100) if (total_pos + total_neg) > 0 else 0
        avg_star = round(sub_df['Rating'].mean(), 2) if 'Rating' in sub_df.columns else 0
        
        m1.metric("亮点总提及", total_pos)
        m2.metric("痛点总提及", total_neg, delta=f"-{total_neg}", delta_color="inverse")
        m3.metric("整体健康度", f"{health_rate}%")
        m4.metric("平均星级评分", f"{avg_star} ⭐")

        # 雷达图
        st.write("")
        col_radar, col_spacer = st.columns([2, 1]) 
        with col_radar:
            fig_radar = go.Figure()
            fig_radar.add_trace(go.Scatterpolar(
                r=analysis_res['满意度'].tolist(),
                theta=analysis_res['维度'].tolist(),
                fill='toself',
                name='满意度 %',
                line_color='#3498db'
            ))
            fig_radar.update_layout(
                polar=dict(radialaxis=dict(visible=True, range=[0, 105])),
                showlegend=False,
                title=f"维度健康度雷达图",
                height=400
            )
            st.plotly_chart(fig_radar, use_container_width=True, key=f"radar_{sub_name}")
        
        # 柱状图 + 折线图
        fig = make_subplots(specs=[[{"secondary_y": True}]])
        
        fig.add_trace(
            go.Bar(name='亮点', x=analysis_res['维度'], y=analysis_res['亮点'], 
                   marker_color='#2ecc71', text=analysis_res['亮点'], textposition='auto'),
            secondary_y=False
        )
        fig.add_trace(
            go.Bar(name='痛点', x=analysis_res['维度'], y=analysis_res['痛点'], 
                   marker_color='#e74c3c', text=analysis_res['痛点'], textposition='auto'),
            secondary_y=False
        )
        fig.add_trace(
            go.Scatter(
                name='满意度 (%)', 
                x=analysis_res['维度'], 
                y=analysis_res['满意度'],
                mode='lines+markers+text', 
                text=analysis_res['满意度'].apply(lambda x: f"{x}%"), 
                textposition="top center", 
                line=dict(color='#3498db', width=3),
                marker=dict(size=8)
            ),
            secondary_y=True 
        )
        fig.add_trace(
            go.Scatter(
                name='维度评分 (1-5)', 
                x=analysis_res['维度'], 
                y=analysis_res['维度评分'],
                mode='lines+markers',
                line=dict(color='#f1c40f', width=2, dash='dot'),
                marker=dict(symbol='star', size=10)
            ),
            secondary_y=True 
        )
        
        fig.update_layout(
            title=f"各维度情感倾向分布与满意度趋势",
            barmode='group',
            height=600,
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
        )
        fig.update_yaxes(title_text="提及次数", secondary_y=False)
        fig.update_yaxes(title_text="满意度分数 (%)", range=[0, 110], secondary_y=True)

        st.plotly_chart(fig, use_container_width=True, key=f"chart_{sub_name}")

        # 竞品弱点
        st.markdown("🔍 **竞品弱点靶向追踪 (Opportunity Analysis)**")
        with st.expander("📊 如何解读机会指数？"):
            st.write("机会指数是我们衡量“竞品失分严重程度”与“市场需求规模”的综合指标。指数越高，代表该维度的“缺口”越大。")
        
        pain_df = analysis_res.sort_values("机会指数", ascending=False).head(3)

        if not pain_df.empty:
            cols = st.columns(3)
            for idx, (_, row) in enumerate(pain_df.iterrows()):
                with cols[idx]:
                    color = "#c0392b" if row['维度评分'] < 3.5 else "#d35400"
                    st.markdown(f"""
                    <div style="padding:15px; border-radius:10px; border-left: 8px solid {color}; 
                                 background-color: #fdfefe; border-top:1px solid #eee; border-right:1px solid #eee;
                                 box-shadow: 2px 2px 8px rgba(0,0,0,0.05); min-height: 200px;">
                        <div style="display:flex; justify-content:space-between;">
                            <h4 style="margin:0;">{row['维度']}</h4>
                            <span style="color:{color}; font-weight:bold;">得分: {row['维度评分']} ⭐</span>
                        </div>
                        <p style="color:gray; font-size:11px; margin-bottom:10px;">
                           机会指数: {row['机会指数']}
                        </p>
                        <p style="font-size:14px;"><b>核心投诉根因：</b><br/>
                        <span style="color:#2c3e50;">{row['痛点分布']}</span></p>
                    </div>
                    """, unsafe_allow_html=True)
        else:
            st.success("✨ 所有维度表现良好，满意度均在 60% 以上！")

        # 词云
        st.markdown("---")
        st.markdown("### ☁️ 原声情感对比词云")
        
        eng_stopwords = set(STOPWORDS)
        custom_garbage = {'marker', 'markers', 'pen', 'pens', 'product', 'really', 'will', 'bought', 'set', 'get', 'much', 'even', 'color', 'paint', 'colors', 'work', 'good', 'great', 'love', 'used', 'using', 'actually', 'amazon', 'br'}
        eng_stopwords.update(custom_garbage)

        pos_df = sub_df[sub_df['Rating'] >= 4.0]
        pos_text = " ".join(pos_df['s_text'].astype(str).str.lower().tolist())
        neg_df = sub_df[sub_df['Rating'] < 4.0]
        neg_text = " ".join(neg_df['s_text'].astype(str).str.lower().tolist())

        col_left, col_right = st.columns(2)

        with col_left:
            st.subheader("🟢 高分区 (4.0-5.0 ⭐)")
            if len(pos_text.strip()) > 30:
                wc_pos = WordCloud(width=500, height=400, background_color='white', colormap='Greens', max_words=50, stopwords=eng_stopwords, collocations=True, random_state=42).generate(pos_text)
                st.image(wc_pos.to_array(), use_container_width=True)
            else:
                st.info("样本量不足")

        with col_right:
            st.subheader("🔴 低分区 (1.0-3.9 ⭐)")
            if len(neg_text.strip()) > 30:
                wc_neg = WordCloud(width=500, height=400, background_color='white', colormap='Reds', max_words=50, stopwords=eng_stopwords, collocations=True, random_state=24).generate(neg_text)
                st.image(wc_neg.to_array(), use_container_width=True)
            else:
                st.success("无明显低分痛点词")

        # 原声溯源
        st.write("")
        with st.expander(f"🔍 深度探查：真实用户评价回溯"):
            target_dim = st.selectbox("选择想要探查的痛点维度:", analysis_res['维度'].tolist(), key=f"select_dim_{sub_name}")
            neg_keywords = []
            if target_dim in FEATURE_DIC:
                for tag, keys in FEATURE_DIC[target_dim].items():
                    if '负面' in tag or '不满' in tag:
                        neg_keywords.extend(keys)
            
            if neg_keywords:
                valid_keys = [re.escape(k) for k in neg_keywords if k.strip()]
                if not valid_keys:
                    st.info("该维度暂无有效的负面关键词。")
                else:
                    search_pattern = '|'.join(valid_keys)
                    vocal_df = sub_df[(sub_df['Rating'] <= 3) & (sub_df['s_text'].str.contains(search_pattern, na=False, flags=re.IGNORECASE))][['Rating', 's_text']].drop_duplicates().head(10)
                    if not vocal_df.empty:
                        st.warning(f"以下是用户在【{target_dim}】维度的真实痛点原声：")
                        for i, (_, row) in enumerate(vocal_df.iterrows()):
                            st.markdown(f"**[{row['Rating']}⭐]** {row['s_text']}")
                            st.divider()
                    else:
                        st.info("该维度下暂未捕捉到高代表性的负面原声评价。")
            else:
                st.write("该维度暂无定义的负面关键词。")

        st.markdown("---")
        
        # --- 深度市场解析 ---
        sub_df = extract_advanced_features(sub_df)
        st.markdown("### 🎯 深度市场深度解析 (Advanced Market Insight)")
        st.markdown("#### 👥 用户画像分布 (Demographic Analysis)")
        
        persona_dim = st.radio("选择画像分析维度:", options=["用户身份", "性别分布", "年龄层次"], horizontal=True, key=f"persona_toggle_{sub_name}")
        dim_map = {"用户身份": "feat_User_Role", "性别分布": "feat_Gender", "年龄层次": "feat_Age_Group"}
        target_col = dim_map[persona_dim]
        
        persona_df = sub_df[(sub_df[target_col].notna()) & (sub_df[target_col] != "未提及") & (sub_df[target_col] != "Unknown")][target_col].value_counts().reset_index()

        if not persona_df.empty:
            fig_pie = go.Figure(data=[go.Pie(labels=persona_df[target_col], values=persona_df['count'], hole=.45, marker=dict(colors=['#636EFA', '#EF553B', '#00CC96', '#AB63FA', '#FFA15A']), textinfo='percent')])
            fig_pie.update_layout(title=dict(text=f"核心访客：{persona_dim}分布", x=0.5, xanchor='center'), height=450, showlegend=True)
            st.plotly_chart(fig_pie, use_container_width=True)
            top_val = persona_df.iloc[0][target_col]
            top_pct = (persona_df.iloc[0]['count'] / persona_df['count'].sum() * 100).round(1)
            st.info(f"📊 **市场洞察：** **{top_val}** 是最主流的群体，占比高达 **{top_pct}%**。")
        else:
            st.warning(f"🔍 暂无明确的 {persona_dim} 维度数据。")

        st.markdown("---")
        st.markdown("#### 🚀 核心痛点维度评分矩阵 (Dynamic Persona-Pain Matrix)")
        
        with st.container():
            st.info("**💡 矩阵说明：** 通过切换下方【身份标签】，可以发现不同人群对产品的不满点是否存在错位。")

        # 确定全局前三维度
        global_top_3 = analysis_res.sort_values("机会指数", ascending=False)['维度'].tolist()[:3]
        while len(global_top_3) < 3:
            global_top_3.append("其他")

        if not analysis_res.empty:
            # --- 核心绘图与表格函数定义 ---
            def draw_sku_bubble_chart(data_source, title_label, suffix, local_dims):
                valid_local = [d for d in local_dims if d and d != "未提及"]
                final_dims = valid_local + [d for d in global_top_3 if d not in valid_local]
                d_x, d_y, d_b = final_dims[0], final_dims[1], final_dims[2]
                
                plot_data = []
                all_skus = data_source['sku_spec'].unique()
                
                for sku in all_skus:
                    sku_df = data_source[data_source['sku_spec'] == sku]
                    
                    def get_metric(target_df, dimension):
                        if dimension == "其他": return 3.0, 0
                        keywords = []
                        if dimension in FEATURE_DIC:
                            for sub_cat in FEATURE_DIC[dimension].values():
                                keywords.extend(sub_cat)
                        if not keywords: return None, 0
                        pat = '|'.join([re.escape(k) for k in keywords if k.strip()])
                        matched = target_df[target_df['s_text'].str.contains(pat, na=False, flags=re.IGNORECASE)]
                        return (matched['Rating'].mean(), len(matched)) if not matched.empty else (None, 0)
                    
                    sc_x, _ = get_metric(sku_df, d_x)
                    sc_y, _ = get_metric(sku_df, d_y)
                    sc_b, _ = get_metric(sku_df, d_b)
                    
                    # 核心过滤：只有在该人群下有维度表现的产品才进入绘图数据集
                    if any(v is not None for v in [sc_x, sc_y, sc_b]):
                        parts = str(sku).split('_')
                        short_name = f"{parts[1]}-{parts[0]}" if len(parts) > 1 else str(sku)
                        
                        plot_data.append({
                            'full_sku': str(sku),
                            'short_name': short_name,
                            'score_x': sc_x if sc_x is not None else 3.0,
                            'score_y': sc_y if sc_y is not None else 3.0,
                            'score_bubble_val': sc_b if sc_b is not None else 3.0 # 这是 1-5 的原始分
                        })
                
                res_df = pd.DataFrame(plot_data)
                
                if res_df.empty:
                    st.warning(f"⚠️ {title_label} 匹配维度下数据量过小，无气泡可显示")
                    return

# --- 1. 绘制气泡图 (颜色综合三个维度评分总和) ---
                fig = go.Figure()
                fig.add_trace(go.Scatter(
                    x=res_df['score_x'], 
                    y=res_df['score_y'], 
                    mode='markers+text',
                    text=res_df['short_name'], 
                    textposition="top center", 
                    # 传入原始数据用于悬停显示
                    customdata=res_df[['full_sku', 'score_bubble_val']], 
                    marker=dict(
                        size=res_df['score_bubble_val'] * 12, # 视觉放大倍数
                        color=res_df['total_sum'],           # 颜色由三维度总和决定
                        cmin=3.0,                            # 固定颜色轴最小值
                        cmax=15.0,                           # 固定颜色轴最大值
                        colorscale='RdYlGn',                 # 红-黄-绿渐变
                        showscale=True, 
                        colorbar=dict(title="综合实力(总分)"),
                        line=dict(width=1, color='DarkSlateGrey')
                    ),
                    # 修正悬停模板，显示原始 1-5 评分
                    hovertemplate = (
                        f"<b>%{{customdata[0]}}</b><br>"
                        f"{d_x}: %{{x:.2f}}<br>"
                        f"{d_y}: %{{y:.2f}}<br>"
                        f"{d_b}: %{{customdata[1]:.2f}}<br>"
                        f"三维度总分: %{{marker.color:.2f}}"
                        f"<extra></extra>"
                    )
                ))
                fig.update_layout(
                    title=f"{title_label}：维度表现分布", 
                    xaxis=dict(title=f"{d_x} 评分", range=[0.8, 5.2]), 
                    yaxis=dict(title=f"{d_y} 评分", range=[0.8, 5.2]), 
                    height=500
                )
                st.plotly_chart(fig, use_container_width=True, key=f"bubble_{suffix}")

                # --- 2. 绘制参数对照表 (数据源严格锁定为 res_df) ---
                st.markdown(f"##### 📋 {title_label} - 图中产品参数明细")
                table_rows = []
                columns_list = ["ASIN", "Brand", "ASP用于", "出墨方式", "线宽", "笔头类型", "支数", "包装材质", "包装方式"]
                
                # 遍历绘图专用的 res_df，确保表格与气泡一一对应
                for _, row in res_df.iterrows():
                    full_sku = row['full_sku']
                    # 使用下划线严格拆分字符串
                    parts = str(full_sku).split('_')
                    row_data = {col: (parts[i].strip() if i < len(parts) else "") for i, col in enumerate(columns_list)}
                    table_rows.append(row_data)
                
                # 隐藏索引渲染表格
                st.dataframe(pd.DataFrame(table_rows), use_container_width=True, hide_index=True)

            # --- 渲染逻辑 ---
            top_roles = sub_df[sub_df['feat_User_Role'] != "未提及"]['feat_User_Role'].value_counts().head(3).index.tolist()
            tab_list = st.tabs(["📊 总体分析"] + [f"👤 {r}" for r in top_roles])
            
            with tab_list[0]:
                draw_sku_bubble_chart(sub_df, "全量数据", f"total_{sub_name}", global_top_3)
            
            for i, role in enumerate(top_roles):
                with tab_list[i+1]:
                    role_sub = sub_df[sub_df['feat_User_Role'] == role]
                    role_neg_text = " ".join(role_sub[role_sub['s_pol'] < 0]['s_text'].astype(str).tolist())
                    dim_counts = {}
                    for dim, mapping in FEATURE_DIC.items():
                        all_keys = [k for sub in mapping.values() for k in sub]
                        count = sum(1 for k in all_keys if k.lower() in role_neg_text.lower())
                        if count > 0: dim_counts[dim] = count
                    role_specific_dims = sorted(dim_counts, key=dim_counts.get, reverse=True)[:3]
                    draw_sku_bubble_chart(role_sub, role, f"role_{i}_{sub_name}", role_specific_dims)
        else:
            st.info("🔍 当前筛选条件下暂无足够的机会维度分析数据。")










