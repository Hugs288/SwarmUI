
let mouseX, mouseY;
let popHide = [];
let lastPopoverTime = 0, lastPopover = null;

class AdvancedPopover {
    /**
     * eg: new AdvancedPopover('my_popover_name', [ { key: 'Button 1', action: () => console.log("Clicked!") } ], true, mouseX, mouseY, document.body, null);
     * Buttons can optionally exclude action to make unclickable.
     */
    constructor(id, buttons, canSearch, x, y, root, preSelect = null, flipYHeight = null, heightLimit = 999999, canSelect = true) {
        this.id = id;
        this.buttons = buttons;
        this.popover = createDiv(`popover_${id}`, 'sui-popover sui_popover_model sui-popover-notransition');
        this.textInput = null;
        this.flipYHeight = flipYHeight;
        this.preSelect = preSelect;
        this.heightLimit = heightLimit;
        this.overExtendBy = 24;
        this.canSelect = canSelect;
        if (canSearch) {
            this.textInput = document.createElement('input');
            this.textInput.type = 'text';
            this.textInput.classList.add('sui_popover_text_input');
            this.textInput.value = '';
            this.textInput.placeholder = 'Search...';
            this.textInput.addEventListener('input', (e) => {
                this.buildList();
                this.optionArea.style.width = (this.optionArea.offsetWidth + this.overExtendBy) + 'px';
            });
            this.textInput.addEventListener('keydown', (e) => {
                this.onKeyDown(e);
            });
            this.popover.appendChild(this.textInput);
        }
        this.optionArea = createDiv(null, 'sui_popover_scrollable_tall');
        this.expectedHeight = 0;
        this.targetY = null;
        this.blockHeight = parseFloat(getComputedStyle(document.documentElement).fontSize) * 1.3;
        this.buildList();
        this.popover.appendChild(this.optionArea);
        root.appendChild(this.popover);
        this.show(x, y);
        if (canSearch) {
            this.textInput.focus();
        }
        this.created = Date.now();
        this.optionArea.style.width = (this.optionArea.offsetWidth + this.overExtendBy) + 'px';
    }

    remove() {
        if (this.popover) {
            this.hide();
            this.popover.remove();
            this.popover = null;
        }
    }

    buildList() {
        let selectedElem = this.selected();
        let selected = this.preSelect ? this.preSelect : selectedElem ? selectedElem.innerText : null;
        let scroll = this.optionArea.scrollTop;
        this.optionArea.innerHTML = '';
        let searchText = this.textInput ? this.textInput.value.toLowerCase() : '';
        let didSelect = false;
        this.expectedHeight = 0;
        this.optionArea.style.width = '';
        for (let button of this.buttons) {
            if ((button.searchable || button.key).toLowerCase().includes(searchText)) {
                let optionDiv = document.createElement(button.href ? 'a' : 'div');
                optionDiv.classList.add('sui_popover_model_button');
                if (button.key_html) {
                    optionDiv.innerHTML = button.key_html;
                }
                else {
                    optionDiv.innerText = button.key;
                }
                if (button.title) {
                    optionDiv.title = button.title;
                }
                if (button.key == selected) {
                    optionDiv.classList.add('sui_popover_model_button_selected');
                    didSelect = true;
                }
                if (button.href) {
                    optionDiv.href = button.href;
                    if (button.is_download) {
                        optionDiv.download = '';
                    }
                }
                else if (!button.action) {
                    optionDiv.classList.add('sui_popover_model_button_disabled');
                }
                else {
                    optionDiv.addEventListener('click', () => {
                        button.action();
                        this.remove();
                    });
                }
                if (button.className) {
                    for (let className of button.className.split(' ')) {
                        optionDiv.classList.add(className);
                    }
                }
                this.optionArea.appendChild(optionDiv);
                this.expectedHeight += this.blockHeight;
            }
        }
        if (!didSelect && this.canSelect) {
            let selected = this.optionArea.querySelector('.sui_popover_model_button');
            if (selected) {
                selected.classList.add('sui_popover_model_button_selected');
            }
        }
        this.optionArea.scrollTop = scroll;
        this.scrollFix();
        if (this.targetY != null) {
            this.reposition();
        }
    }

    selected() {
        if (!this.popover) {
            return [];
        }
        return this.popover.getElementsByClassName('sui_popover_model_button_selected')[0];
    }

    scrollFix() {
        let selected = this.selected();
        if (!selected) {
            return;
        }
        if (selected.offsetTop + selected.offsetHeight > this.optionArea.scrollTop + this.optionArea.offsetHeight) {
            this.optionArea.scrollTop = selected.offsetTop + selected.offsetHeight - this.optionArea.offsetHeight + 6;
        }
        else if (selected.offsetTop < this.optionArea.scrollTop) {
            this.optionArea.scrollTop = selected.offsetTop;
        }
    }

    possible() {
        if (!this.popover) {
            return [];
        }
        return [...this.popover.getElementsByClassName('sui_popover_model_button')].filter(e => !e.classList.contains('sui_popover_model_button_disabled'));
    }

    onKeyDown(e) {
        if (e.shiftKey || e.ctrlKey) {
            return true;
        }
        let possible = this.possible();
        if (!possible) {
            return true;
        }
        if (e.key == 'Escape') {
            this.remove();
        }
        else if (e.key == 'Tab' || e.key == 'Enter') {
            let selected = this.popover.querySelector('.sui_popover_model_button_selected');
            if (selected) {
                this.hide();
                selected.click();
            }
            e.preventDefault();
            e.stopPropagation();
            return false;
        }
        else if (e.key == 'ArrowUp') {
            let selectedIndex = possible.findIndex(e => e.classList.contains('sui_popover_model_button_selected'));
            if (selectedIndex == -1) {
                selectedIndex = 0;
            }
            possible[selectedIndex].classList.remove('sui_popover_model_button_selected');
            possible[(selectedIndex + possible.length - 1) % possible.length].classList.add('sui_popover_model_button_selected');
            this.scrollFix();
        }
        else if (e.key == 'ArrowDown') {
            let selectedIndex = possible.findIndex(e => e.classList.contains('sui_popover_model_button_selected'));
            if (selectedIndex == -1) {
                selectedIndex = 0;
            }
            possible[selectedIndex].classList.remove('sui_popover_model_button_selected');
            possible[(selectedIndex + 1) % possible.length].classList.add('sui_popover_model_button_selected');
            this.scrollFix();
        }
        else {
            return true;
        }
        e.preventDefault();
        return false;
    }

    hide() {
        if (this.popover.dataset.visible == "true") {
            this.popover.classList.remove('sui-popover-visible');
            this.popover.dataset.visible = "false";
            popHide.splice(popHide.indexOf(this), 1);
        }
    }

    isHidden() {
        return this.popover.dataset.visible != "true";
    }

    reposition() {
        if (this.popover.classList.contains('sui_popover_reverse')) {
            this.popover.classList.remove('sui_popover_reverse');
        }
        let y;
        let maxHeight;
        let extraHeight = (this.textInput ? this.textInput.offsetHeight : 0) + 32;
        let rawExpected = Math.min(this.expectedHeight, this.heightLimit);
        let expected = rawExpected + extraHeight;
        if (this.targetY + expected < window.innerHeight) {
            y = this.targetY;
            maxHeight = rawExpected;
        }
        else if (this.flipYHeight != null && this.targetY > window.innerHeight / 2) {
            y = Math.max(0, this.targetY - this.flipYHeight - expected);
            this.popover.classList.add('sui_popover_reverse');
            maxHeight = Math.min(this.targetY - this.flipYHeight - 32, rawExpected);
        }
        else {
            y = this.targetY;
            maxHeight = window.innerHeight - y - extraHeight - 10;
        }
        this.popover.style.top = `${y}px`;
        this.optionArea.style.maxHeight = `${maxHeight}px`;
    }

    show(targetX, targetY) {
        this.targetY = targetY;
        if (this.popover.dataset.visible == "true") {
            this.hide();
        }
        this.popover.classList.add('sui-popover-visible');
        this.popover.style.width = '200px';
        this.popover.dataset.visible = "true";
        let x = Math.min(targetX, window.innerWidth - this.popover.offsetWidth - 10);
        let y = Math.min(targetY, window.innerHeight - this.popover.offsetHeight);
        this.popover.style.left = `${x}px`;
        this.popover.style.top = `${y}px`;
        this.popover.style.width = '';
        this.reposition();
        popHide.push(this);
        lastPopoverTime = Date.now();
        lastPopover = this;
    }
}

class UIImprovementHandler {
    constructor() {
        this.lastPopover = null;
        this.lastShift = false;
        this.lastSelectedTextbox = null;
        this.timeOfLastTextboxSelectTrack = 0;
        this.lastTextboxCursorPos = -1;
        document.addEventListener('focusout', (e) => {
            if (e.target.tagName == 'TEXTAREA') {
                this.lastSelectedTextbox = e.target;
                this.timeOfLastTextboxSelectTrack = Date.now();
                this.lastTextboxCursorPos = e.target.selectionEnd;
            }
        }, true);
        document.addEventListener('mousedown', (e) => {
            this.lastShift = e.shiftKey;
            if (e.target.tagName == 'SELECT') {
                if (!this.lastShift && this.shouldAlterSelect(e.target)) {
                    e.preventDefault();
                    e.stopPropagation();
                    return false;
                }
            }
        }, true);
        document.addEventListener('click', (e) => {
            if (e.target.tagName == 'SELECT' && !this.lastShift && this.shouldAlterSelect(e.target)) { // e.shiftKey doesn't work in click for some reason
                // The tiny delay is to try to fight broken browser extensions that spazz out when elements are spawned from a click
                // (eg 1Password, Eno Capital One, iCloud Passwords are known offenders)
                setTimeout(() => {
                    this.onSelectClicked(e.target, e);
                }, 1);
                e.preventDefault();
                e.stopPropagation();
                return false;
            }
        }, true);
        document.addEventListener('mouseup', (e) => {
            if (e.target.tagName == 'SELECT' && !e.shiftKey && this.shouldAlterSelect(e.target)) {
                e.preventDefault();
                e.stopPropagation();
                return false;
            }
        }, true);
        function updateVal(input, newVal, step) {
            let min = parseFloat(input.min);
            if (typeof min == 'number' && !isNaN(min)) {
                newVal = Math.max(newVal, min);
            }
            let max = parseFloat(input.max);
            if (typeof max == 'number' && !isNaN(max)) {
                newVal = Math.min(newVal, max);
            }
            input.value = roundToStrAuto(newVal, step);
            triggerChangeFor(input);
        }
        window.addEventListener('wheel', (e) => {
            if (!e.target || !e.target.matches(':focus')) {
                return;
            }
            if (e.target.tagName == 'INPUT' && (e.target.type == 'number' || e.target.type == 'range')) {
                let input = e.target;
                let step = parseFloat(input.step);
                if (typeof step != 'number' || isNaN(step)) {
                    step = 1;
                }
                let value = parseFloat(input.value) || 0;
                if (e.deltaY > 0) {
                    updateVal(input, value - step, step);
                }
                else if (e.deltaY < 0) {
                    updateVal(input, value + step, step);
                }
                e.preventDefault();
                e.stopPropagation();
                return false;
            }
        }, {capture:true, passive:false});
        let lastX = 0, lastY = 0;
        let stepDist = 10;
        let clickedElem = null;
        window.addEventListener('mousemove', (e) => {
            if (clickedElem) {
                if (e.buttons != 1) {
                    clickedElem.style.cursor = '';
                    return;
                }
                clickedElem.style.cursor = 'ew-resize';
                if (lastX == 0 && lastY == 0) {
                    lastX = e.pageX;
                    lastY = e.pageY;
                    return;
                }
                let moveX = e.pageX - lastX;
                let moveY = e.pageY - lastY;
                if (Math.abs(moveX) < stepDist && Math.abs(moveY) < stepDist) {
                    return;
                }
                moveX = Math.round(moveX / stepDist);
                moveY = Math.round(moveY / stepDist);
                lastX = e.pageX;
                lastY = e.pageY;
                let step = parseFloat(clickedElem.step);
                if (typeof step != 'number' || isNaN(step)) {
                    step = 1;
                }
                let value = parseFloat(clickedElem.value) || 0;
                let newVal = value + (moveX - moveY) * step;
                updateVal(clickedElem, newVal, step);
                e.preventDefault();
                e.stopPropagation();
                return false;
            }
        }, {capture:true, passive:false});
        window.addEventListener('mousedown', (e) => {
            clickedElem = null;
            if (e.target.tagName == 'INPUT' && e.target.type == 'number') {
                lastX = 0;
                lastY = 0;
                clickedElem = e.target;
            }
        }, true);
        window.addEventListener('mouseup', (e) => {
            clickedElem = null;
            if (e.target.tagName == 'INPUT' && e.target.type == 'number') {
                e.target.style.cursor = '';
                lastX = 0;
                lastY = 0;
            }
        }, true);
        let isDoingADrag = false;
        document.addEventListener('dragenter', (e) => {
            if (isDoingADrag) {
                return;
            }
            isDoingADrag = true;
            let files = this.getFileList(e.dataTransfer, e);
            if (files.length > 0 && files.filter(f => f.type.startsWith('image/')).length > 0) {
                let targets = document.getElementsByClassName('drag_image_target');
                for (let target of targets) {
                    target.classList.add('drag_image_target_highlight');
                }
            }
        }, true);
        function clearDrag() {
            setTimeout(() => {
                isDoingADrag = false;
                let targets = document.getElementsByClassName('drag_image_target'); // intentionally don't search "_highlight" due to browse misbehavior
                for (let target of targets) {
                    target.classList.remove('drag_image_target_highlight');
                }
            }, 1);
        }
        document.addEventListener('drop', (e) => {
            clearDrag();
        }, true);
        document.addEventListener('dragleave', (e) => {
            if (e.clientY <= 0 || e.clientX <= 0 || e.clientX >= window.innerWidth || e.clientY >= window.innerHeight) {
                clearDrag();
            }
        }, true);
        window.addEventListener('mousemove', (e) => {
            if (isDoingADrag && e.buttons == 0) {
                clearDrag();
            }
        }, true);
    }

    /** Returns a list of files from the given dataTransfer object, auto-correcting for browsers inconsistently handling certain drag types. */
    getFileList(dataTransfer, e) {
        if (!dataTransfer) {
            return [];
        }
        let files = dataTransfer.files;
        if (!files || !files.length) {
            files = [...dataTransfer.items || []].filter(item => item.kind == "file");
        }
        if (!files.length) {
            let uris = dataTransfer.getData('text/uri-list');
            if (uris) {
                files = uris.split('\n');
            }
            files = files.map(f => new File([f], f, {type: guessMimeTypeForExtension(f)}));
        }
        if (!files.length && e && e.srcElement) {
            let img = e.srcElement;
            if (img.tagName == 'IMG') {
                files = [new File([img.src], img.src, {type: guessMimeTypeForExtension(img.src)})];
            }
        }
        return files;
    }

    getLastSelectedTextbox() {
        let now = Date.now();
        if (now - this.timeOfLastTextboxSelectTrack > 1000) {
            return [null, -1];
        }
        return [this.lastSelectedTextbox, this.lastTextboxCursorPos];
    }

    shouldAlterSelect(elem) {
        if (elem.options.length > 1) {
            return true;
        }
        if ([... elem.options].filter(o => o.innerText.includes('(')).length > 0) {
            return true;
        }
        return false;
    }

    onSelectClicked(elem, e) {
        if (this.lastPopover && this.lastPopover.popover) {
            this.lastPopover.remove();
            this.lastPopover = null;
            e.preventDefault();
            e.stopPropagation();
            return false;
        }
        let popId = `uiimprover_${elem.id}`;
        let rect = elem.getBoundingClientRect();
        let buttons = [...elem.options].filter(o => o.style.display != 'none').map(o => { return { key_html: o.dataset.cleanname, title: o.title, key: o.innerText, searchable: `${o.dataset.cleanname} ${o.innerText} ${o.value}`, action: () => { elem.value = o.value; triggerChangeFor(elem); } }; })
        this.lastPopover = new AdvancedPopover(popId, buttons, true, rect.x, rect.y, elem.parentElement, elem.selectedIndex < 0 ? null : elem.selectedOptions[0].innerText, 0);
        e.preventDefault();
        e.stopPropagation();
        return false;
    }

    /** This used to be a CSS Animation, but browsers try so hard to make those pretty and smooth that it makes a noticeable GPU perf impact. Ow. */
    runLoadSpinner(div) {
        if (div.dataset.is_spinner_going) {
            return;
        }
        div.dataset.is_spinner_going = 'true';
        setTimeout(() => {
            let s1 = div.querySelector('.loadspin1');
            if (!s1) {
                return;
            }
            let s2 = div.querySelector('.loadspin2');
            let s3 = div.querySelector('.loadspin3');
            let interval;
            let time = 0;
            let step = 0.05;
            interval = setInterval(() => {
                if (!div.isConnected || div.style.display == 'none' || !s1) {
                    clearInterval(interval);
                    delete div.dataset.is_spinner_going;
                    return;
                }
                time += step;
                s1.style.transform = `rotate(${((time - 0.45) / 1.2) * 360}deg)`;
                s2.style.transform = `rotate(${((time - 0.3) / 1.2) * 360}deg)`;
                s3.style.transform = `rotate(${((time - 0.15) / 1.2) * 360}deg)`;
            }, step * 1000);
        }, 100);
    }
}

uiImprover = new UIImprovementHandler();

///////////// Older-style popover code, to be cleaned

function doPopHideCleanup(target) {
    for (let x = 0; x < popHide.length; x++) {
        let id = popHide[x];
        let pop = id.popover ? id.popover : getRequiredElementById(`popover_${id}`);
        if (id == lastPopover && Date.now() - lastPopoverTime < 50) {
            continue;
        }
        if (pop.contains(target) && !target.classList.contains('sui_popover_model_button')) {
            continue;
        }
        if (id instanceof AdvancedPopover) {
            if (Date.now() - id.created > 50) {
                id.remove();
            }
        }
        else {
            pop.classList.remove('sui-popover-visible');
            pop.dataset.visible = "false";
            popHide.splice(x, 1);
        }
    }
}

document.addEventListener('mousedown', (e) => {
    mouseX = e.pageX;
    mouseY = e.pageY;
    if (e.button == 2) { // right-click
        doPopHideCleanup(e.target);
    }
}, true);

document.addEventListener('click', (e) => {
    if (e.target.tagName == 'BODY') {
        return; // it's impossible on the genpage to actually click body, so this indicates a bugged click, so ignore it
    }
    doPopHideCleanup(e.target);
}, true);

/** Ensures the popover for the given ID is hidden. */
function hidePopover(id) {
    let pop = getRequiredElementById(`popover_${id}`);
    if (pop.dataset.visible == "true") {
        pop.classList.remove('sui-popover-visible');
        pop.dataset.visible = "false";
        popHide.splice(popHide.indexOf(id), 1);
    }
}

/** Shows the given popover, optionally at the specified location. */
function showPopover(id, targetX = mouseX, targetY = mouseY) {
    let pop = getRequiredElementById(`popover_${id}`);
    if (pop.dataset.visible == "true") {
        hidePopover(id); // Hide to reset before showing again.
    }
    pop.classList.add('sui-popover-visible');
    pop.style.width = '200px';
    pop.dataset.visible = "true";
    let x = Math.min(targetX, window.innerWidth - pop.offsetWidth - 10);
    let y = Math.min(targetY, window.innerHeight - pop.offsetHeight);
    pop.style.left = `${x}px`;
    pop.style.top = `${y}px`;
    pop.style.width = '';
    popHide.push(id);
    lastPopoverTime = Date.now();
    lastPopover = id;
}

/** Toggles the given popover, showing it or hiding it as relevant. */
function doPopover(id, e) {
    let pop = getRequiredElementById(`popover_${id}`);
    if (pop.dataset.visible == "true") {
        hidePopover(id);
    }
    else if (e && e.target) {
        let rect = e.target.getBoundingClientRect();
        showPopover(id, rect.left, rect.bottom + 15);
    }
    else {
        showPopover(id);
    }
    if (e) {
        e.preventDefault();
        e.stopImmediatePropagation();
    }
}

/** Shows a notice popover with the given text and color. */
function doNoticePopover(text, className, targetX = mouseX, targetY = mouseY) {
    let pop = createDiv(null, `sui-popover sui_popover_model ${className} sui-popover-notice`);
    pop.style.width = '200px';
    let x = Math.min(targetX, window.innerWidth - pop.offsetWidth - 10);
    let y = Math.min(targetY, window.innerHeight - pop.offsetHeight);
    pop.style.left = `${x}px`;
    pop.style.top = `${y}px`;
    pop.style.width = '';
    pop.innerText = translate(text);
    document.body.appendChild(pop);
    setTimeout(() => {
        pop.classList.add('sui-popover-notice-fade-1s');
        setTimeout(() => {
            pop.remove();
        }, 1500);
    }, 1000);
}
