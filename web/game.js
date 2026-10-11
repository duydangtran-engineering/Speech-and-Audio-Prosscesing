// Voice-controlled maze game.
//
// The player never stops moving. A spoken direction is queued as an
// intention and applied at the next point where that turn is legal.
// That is what makes ~200 ms of recognition latency invisible: the
// command is a plan, not a keypress.

export const TILE = 30;

// Junctions sit four tiles apart, so there are always three tiles of
// corridor to travel before the next decision point. Spoken commands
// need that room: on a maze where every tile is a junction, the turn
// lands wherever the recogniser happened to finish, which feels random.
// # wall, . coin, P player start, E enemy start
const MAZE = [
  '#######################',
  '#E....###............E#',
  '#.###.#######.###.###.#',
  '#.###.#######.###.###.#',
  '#.###.#######.###.###.#',
  '#.............###.....#',
  '#####.###.###.#######.#',
  '#####.###.###.#######.#',
  '#####.###.###.#######.#',
  '#.###....P........###.#',
  '#.#######.#######.###.#',
  '#.#######.#######.###.#',
  '#.#######.#######.###.#',
  '#.........###.........#',
  '#.###.#######.###.###.#',
  '#.###.#######.###.###.#',
  '#.###.#######.###.###.#',
  '#E....###.........###E#',
  '#######################',
];

export const DIRECTIONS = {
  len:   { x: 0, y: -1, name: 'lên' },
  xuong: { x: 0, y: 1, name: 'xuống' },
  trai:  { x: -1, y: 0, name: 'trái' },
  phai:  { x: 1, y: 0, name: 'phải' },
};

const ROUND_SECONDS = 60;
const PLAYER_SPEED = 3.0;      // tiles per second
const ENEMY_SPEED = 2.1;
const TURN_TOLERANCE = 0.22;   // how near a tile centre a turn may start
const SHIELD_DURATION = 2.0;
const SHIELD_COOLDOWN = 8.0;
const DASH_DURATION = 1.0;
const DASH_COOLDOWN = 6.0;
const DASH_MULTIPLIER = 2.8;
const HIT_PENALTY = 50;
const HIT_INVULNERABLE = 1.5;

const COLORS = {
  floor: '#fbfcff',
  wall: '#dbe2f7',
  wallEdge: '#c3cdec',
  coin: '#f0a500',
  coinRing: '#fde8b8',
  player: '#4f46e5',
  playerDash: '#0ea5e9',
  enemy: '#f43f5e',
  enemyDark: '#be123c',
  shield: '#10b981',
  pending: '#f59e0b',
  ink: '#1e2140',
};

export class Game {
  constructor(canvas) {
    this.canvas = canvas;
    this.ctx = canvas.getContext('2d');
    this.cols = MAZE[0].length;
    this.rows = MAZE.length;

    // Render at device resolution so the maze stays crisp on high-DPI
    // screens, then scale back down with CSS.
    const dpr = Math.min(window.devicePixelRatio || 1, 2);
    canvas.width = this.cols * TILE * dpr;
    canvas.height = this.rows * TILE * dpr;
    canvas.style.width = `${this.cols * TILE}px`;
    canvas.style.height = `${this.rows * TILE}px`;
    this.ctx.scale(dpr, dpr);

    this.onEvent = () => {};
    this.reset();
  }

  reset() {
    this.walls = [];
    this.coins = [];
    this.enemySpawns = [];
    let start = { x: 1, y: 1 };

    for (let y = 0; y < this.rows; y++) {
      this.walls[y] = [];
      this.coins[y] = [];
      for (let x = 0; x < this.cols; x++) {
        const cell = MAZE[y][x];
        this.walls[y][x] = cell === '#';
        this.coins[y][x] = cell === '.';
        if (cell === 'P') start = { x, y };
        if (cell === 'E') this.enemySpawns.push({ x, y });
      }
    }

    this.player = { x: start.x, y: start.y,
                    dir: DIRECTIONS.phai, pending: null };
    this.enemies = this.enemySpawns.slice(0, 4).map((spawn, i) => ({
      x: spawn.x, y: spawn.y,
      dir: Object.values(DIRECTIONS)[i % 4],
      delay: 2.0 + i * 2.5,     // staggered release, so the start is fair
    }));

    this.score = 0;
    this.timeLeft = ROUND_SECONDS;
    this.shield = { active: 0, cooldown: 0, duration: SHIELD_DURATION,
                    cooldownFull: SHIELD_COOLDOWN };
    this.dash = { active: 0, cooldown: 0, duration: DASH_DURATION,
                  cooldownFull: DASH_COOLDOWN };
    this.invulnerable = 0;
    this.running = false;
    this.finished = false;
    this.coinsTotal = this.coins.flat().filter(Boolean).length;
    this.coinsLeft = this.coinsTotal;
    this.flash = null;
    this.draw();
  }

  isWall(x, y) {
    if (y < 0 || y >= this.rows || x < 0 || x >= this.cols) return true;
    return this.walls[y][x];
  }

  /** A spoken or typed command. Returns a short description for the HUD. */
  command(label) {
    if (!this.running) return null;

    if (DIRECTIONS[label]) {
      this.player.pending = DIRECTIONS[label];
      return DIRECTIONS[label].name;
    }
    if (label === 'khien') {
      if (this.shield.cooldown > 0) return 'khiên chưa hồi';
      this.shield.active = SHIELD_DURATION;
      this.shield.cooldown = SHIELD_COOLDOWN;
      this.flash = { text: 'KHIÊN', time: 1.0, color: COLORS.shield };
      return 'khiên';
    }
    if (label === 'luot') {
      if (this.dash.cooldown > 0) return 'lướt chưa hồi';
      this.dash.active = DASH_DURATION;
      this.dash.cooldown = DASH_COOLDOWN;
      this.flash = { text: 'LƯỚT', time: 1.0, color: COLORS.playerDash };
      return 'lướt';
    }
    return null;
  }

  start() {
    this.reset();
    this.running = true;
    this.last = performance.now();
    requestAnimationFrame(this.loop.bind(this));
  }

  loop(now) {
    const dt = Math.min((now - this.last) / 1000, 0.05);
    this.last = now;
    if (this.running) this.update(dt);
    this.draw();
    if (this.running) requestAnimationFrame(this.loop.bind(this));
  }

  update(dt) {
    this.timeLeft -= dt;
    if (this.timeLeft <= 0 || this.coinsLeft === 0) {
      this.timeLeft = Math.max(0, this.timeLeft);
      this.running = false;
      this.finished = true;
      this.onEvent({ type: 'end', score: this.score,
                     cleared: this.coinsLeft === 0 });
      return;
    }

    for (const timer of [this.shield, this.dash]) {
      timer.active = Math.max(0, timer.active - dt);
      timer.cooldown = Math.max(0, timer.cooldown - dt);
    }
    this.invulnerable = Math.max(0, this.invulnerable - dt);
    if (this.flash) {
      this.flash.time -= dt;
      if (this.flash.time <= 0) this.flash = null;
    }

    const speed = PLAYER_SPEED * (this.dash.active > 0 ? DASH_MULTIPLIER : 1);
    this.moveEntity(this.player, speed, dt, true);

    for (const enemy of this.enemies) {
      if (enemy.delay > 0) { enemy.delay -= dt; continue; }
      this.moveEntity(enemy, ENEMY_SPEED, dt, false);
      if (Math.hypot(enemy.x - this.player.x, enemy.y - this.player.y) < 0.7) {
        this.hit();
      }
    }

    const cx = Math.round(this.player.x), cy = Math.round(this.player.y);
    if (Math.abs(this.player.x - cx) < 0.35 && Math.abs(this.player.y - cy) < 0.35
        && this.coins[cy]?.[cx]) {
      this.coins[cy][cx] = false;
      this.coinsLeft--;
      this.score += 10;
    }
  }

  hit() {
    if (this.shield.active > 0 || this.invulnerable > 0
        || this.dash.active > 0) return;
    this.score = Math.max(0, this.score - HIT_PENALTY);
    this.invulnerable = HIT_INVULNERABLE;
    this.flash = { text: '-50', time: 0.8, color: COLORS.enemy };
    this.onEvent({ type: 'hit' });
  }

  /**
   * Move along the grid. The pending turn is taken as soon as the entity
   * is close enough to a tile centre for that turn to be legal.
   */
  moveEntity(entity, speed, dt, isPlayer) {
    // Turning back the way you came needs no junction, so it happens the
    // instant the command lands. At least one direction always answers
    // immediately, which is most of what makes the controls feel alive.
    if (isPlayer && entity.pending) {
      const p = entity.pending;
      if (p.x === -entity.dir.x && p.y === -entity.dir.y) {
        entity.dir = p;
        entity.pending = null;
      }
    }

    const cx = Math.round(entity.x), cy = Math.round(entity.y);
    const centred = Math.abs(entity.x - cx) < TURN_TOLERANCE
                 && Math.abs(entity.y - cy) < TURN_TOLERANCE;

    if (centred) {
      if (isPlayer && entity.pending) {
        const p = entity.pending;
        if (!this.isWall(cx + p.x, cy + p.y)) {
          entity.dir = p;
          entity.pending = null;
          entity.x = cx; entity.y = cy;
        }
      }
      if (!isPlayer) this.chooseEnemyDirection(entity, cx, cy);

      if (this.isWall(cx + entity.dir.x, cy + entity.dir.y)) {
        entity.x = cx; entity.y = cy;
        if (!isPlayer) this.chooseEnemyDirection(entity, cx, cy, true);
        else return;   // idle against the wall until a new command arrives
      }
    }

    entity.x += entity.dir.x * speed * dt;
    entity.y += entity.dir.y * speed * dt;
  }

  chooseEnemyDirection(enemy, cx, cy, force = false) {
    const options = Object.values(DIRECTIONS).filter(
      d => !this.isWall(cx + d.x, cy + d.y));
    if (!options.length) return;

    const forward = options.filter(
      d => !(d.x === -enemy.dir.x && d.y === -enemy.dir.y));
    const usable = forward.length ? forward : options;

    const atJunction = usable.length > 1;
    if (!force && !atJunction) { enemy.dir = usable[0]; return; }

    // Chase most of the time, wander sometimes, so the enemies do not
    // collapse into one unavoidable line.
    if (Math.random() < 0.65) {
      usable.sort((a, b) =>
        Math.hypot(cx + a.x - this.player.x, cy + a.y - this.player.y)
        - Math.hypot(cx + b.x - this.player.x, cy + b.y - this.player.y));
      enemy.dir = usable[0];
    } else {
      enemy.dir = usable[Math.floor(Math.random() * usable.length)];
    }
  }

  // --- drawing --------------------------------------------------------
  roundRect(x, y, w, h, r) {
    const ctx = this.ctx;
    ctx.beginPath();
    ctx.moveTo(x + r, y);
    ctx.arcTo(x + w, y, x + w, y + h, r);
    ctx.arcTo(x + w, y + h, x, y + h, r);
    ctx.arcTo(x, y + h, x, y, r);
    ctx.arcTo(x, y, x + w, y, r);
    ctx.closePath();
  }

  draw() {
    const ctx = this.ctx;
    const width = this.cols * TILE, height = this.rows * TILE;

    ctx.fillStyle = COLORS.floor;
    ctx.fillRect(0, 0, width, height);

    for (let y = 0; y < this.rows; y++) {
      for (let x = 0; x < this.cols; x++) {
        if (this.walls[y][x]) {
          ctx.fillStyle = COLORS.wall;
          this.roundRect(x * TILE + 1, y * TILE + 1, TILE - 2, TILE - 2, 6);
          ctx.fill();
          ctx.strokeStyle = COLORS.wallEdge;
          ctx.lineWidth = 1;
          ctx.stroke();
        } else if (this.coins[y][x]) {
          const px = x * TILE + TILE / 2, py = y * TILE + TILE / 2;
          ctx.fillStyle = COLORS.coinRing;
          ctx.beginPath(); ctx.arc(px, py, 6, 0, Math.PI * 2); ctx.fill();
          ctx.fillStyle = COLORS.coin;
          ctx.beginPath(); ctx.arc(px, py, 3.2, 0, Math.PI * 2); ctx.fill();
        }
      }
    }

    for (const enemy of this.enemies) {
      if (enemy.delay > 0) continue;
      const px = enemy.x * TILE + TILE / 2, py = enemy.y * TILE + TILE / 2;
      const r = TILE * 0.24;
      ctx.fillStyle = 'rgba(30,33,64,0.10)';
      ctx.beginPath(); ctx.ellipse(px, py + r * 0.9, r * 0.9, r * 0.35, 0, 0, 6.3);
      ctx.fill();
      ctx.fillStyle = COLORS.enemy;
      ctx.beginPath(); ctx.arc(px, py, r, 0, Math.PI * 2); ctx.fill();
      ctx.fillStyle = '#ffffff';
      ctx.beginPath();
      ctx.arc(px - r * 0.35, py - r * 0.15, r * 0.22, 0, Math.PI * 2);
      ctx.arc(px + r * 0.35, py - r * 0.15, r * 0.22, 0, Math.PI * 2);
      ctx.fill();
      ctx.fillStyle = COLORS.enemyDark;
      ctx.beginPath();
      ctx.arc(px - r * 0.35, py - r * 0.1, r * 0.1, 0, Math.PI * 2);
      ctx.arc(px + r * 0.35, py - r * 0.1, r * 0.1, 0, Math.PI * 2);
      ctx.fill();
    }

    const px = this.player.x * TILE + TILE / 2;
    const py = this.player.y * TILE + TILE / 2;
    const pr = TILE * 0.26;

    if (this.invulnerable > 0 && Math.floor(this.invulnerable * 10) % 2 === 0) {
      ctx.globalAlpha = 0.35;
    }
    ctx.fillStyle = 'rgba(30,33,64,0.12)';
    ctx.beginPath();
    ctx.ellipse(px, py + pr, pr, pr * 0.35, 0, 0, 6.3);
    ctx.fill();
    ctx.fillStyle = this.dash.active > 0 ? COLORS.playerDash : COLORS.player;
    ctx.beginPath(); ctx.arc(px, py, pr, 0, Math.PI * 2); ctx.fill();
    ctx.fillStyle = 'rgba(255,255,255,0.85)';
    ctx.beginPath();
    ctx.arc(px + this.player.dir.x * pr * 0.3,
            py + this.player.dir.y * pr * 0.3, pr * 0.3, 0, Math.PI * 2);
    ctx.fill();
    ctx.globalAlpha = 1;

    if (this.shield.active > 0) {
      ctx.strokeStyle = COLORS.shield;
      ctx.lineWidth = 2.5;
      ctx.globalAlpha = 0.5 + 0.5 * Math.sin(performance.now() / 140);
      ctx.beginPath(); ctx.arc(px, py, pr * 1.7, 0, Math.PI * 2); ctx.stroke();
      ctx.globalAlpha = 1;
    }

    // The queued turn, shown as an arrow. The command has been heard; it
    // is waiting for a junction. Making the wait visible turns "it
    // ignored me" into "it understood, it is about to turn".
    if (this.player.pending) {
      const p = this.player.pending;
      ctx.save();
      ctx.translate(px + p.x * TILE * 0.8, py + p.y * TILE * 0.8);
      ctx.rotate(Math.atan2(p.y, p.x));
      ctx.fillStyle = COLORS.pending;
      ctx.globalAlpha = 0.5 + 0.5 * Math.sin(performance.now() / 130);
      ctx.beginPath();
      ctx.moveTo(8, 0); ctx.lineTo(-5, -6); ctx.lineTo(-5, 6);
      ctx.closePath(); ctx.fill();
      ctx.restore();
      ctx.globalAlpha = 1;
    }

    if (this.flash) {
      ctx.globalAlpha = Math.min(1, this.flash.time);
      ctx.fillStyle = this.flash.color ?? COLORS.ink;
      ctx.font = '700 20px "Segoe UI", system-ui, sans-serif';
      ctx.textAlign = 'center';
      ctx.fillText(this.flash.text, px, py - TILE * 0.9);
      ctx.globalAlpha = 1;
      ctx.textAlign = 'start';
    }

    if (this.finished) {
      ctx.fillStyle = 'rgba(251,252,255,0.92)';
      ctx.fillRect(0, 0, width, height);
      ctx.textAlign = 'center';
      ctx.fillStyle = COLORS.ink;
      ctx.font = '700 40px "Segoe UI", system-ui, sans-serif';
      ctx.fillText(this.coinsLeft === 0 ? 'HOÀN THÀNH' : 'HẾT GIỜ',
                   width / 2, height / 2 - 16);
      ctx.fillStyle = COLORS.player;
      ctx.font = '600 26px "Segoe UI", system-ui, sans-serif';
      ctx.fillText(`${this.score} điểm`, width / 2, height / 2 + 24);
      ctx.fillStyle = '#6b7089';
      ctx.font = '400 15px "Segoe UI", system-ui, sans-serif';
      ctx.fillText(`Ăn được ${this.coinsTotal - this.coinsLeft}/${this.coinsTotal} xu`,
                   width / 2, height / 2 + 54);
      ctx.textAlign = 'start';
    }
  }
}
